import chainer
import chainer.functions as F
import chainer.links as L
import six

from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


class StraightThrough(chainer.function_node.FunctionNode):
    def check_type_forward(self, in_types):
        n_in = in_types.size()
        chainer.utils.type_check.expect(2 == n_in)
        x_type, w_type = in_types

        chainer.utils.type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 3,
            x_type.ndim <= 4,
            w_type.ndim == 2,
            x_type.shape[1] == w_type.shape[1],
            )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xs = inputs[0]
        W = inputs[1]
        xp = chainer.cuda.get_array_module(*inputs)
        e = W

        if not chainer.utils.type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(W): {0}, type(x): {1}'
                             .format(type(W), type(xs)))

        # broadcast to calculate l2 norm
        xs = xp.expand_dims(xs, 1)
        shape = list(xs.shape)
        shape[1] = W.shape[0]
        xs = xp.broadcast_to(xs, tuple(shape))

        if xs.ndim == 5:
            W = xp.broadcast_to(
                xp.reshape(W, (1,) + W.shape + (1, 1)), xs.shape)
        elif xs.ndim == 4:
            W = xp.broadcast_to(
                xp.reshape(W, (1,) + W.shape + (1,)), xs.shape)

        # get index of minimum l2 norm
        self.indexes = xp.argmin(
            xp.sum((xs - W) ** 2, axis=2), axis=1).astype(xp.int32)

        # quantize
        embeded = e[self.indexes]
        if embeded.ndim == 4:
            embeded = embeded.transpose((0, 3, 1, 2))
        elif embeded.ndim == 3:
            embeded = embeded.transpose((0, 2, 1))
        return embeded,

    def backward(self, indexes, grad_outputs):
        xs, W = self.get_retained_inputs()
        gy, = grad_outputs
        ret = []

        if 0 in indexes:
            ret.append(gy)
        if 1 in indexes:
            xp = chainer.cuda.get_array_module(*grad_outputs)
            if gy.ndim == 4:
                gy = gy.transpose((0, 2, 3, 1))
            elif gy.ndim == 3:
                gy = gy.transpose((0, 2, 1))
            gy = gy.reshape((-1, gy.shape[-1]))
            self.indexes = xp.eye(W.shape[0])[self.indexes.reshape((-1))]
            gW = self.indexes.T.dot(gy.data).astype(gy.dtype)
            gW = chainer.Variable(gW)
            ret.append(gW)
        return ret


def straight_through(x, W):
    y, = StraightThrough().apply((x, W))
    return y


class VQ(chainer.link.Link):
    def __init__(self, k, d=None, initialW=None):
        super(VQ, self).__init__()
        self.k = k
        with self.init_scope():
            W_initializer = chainer.initializers._get_initializer(initialW)
            self.W = chainer.variable.Parameter(W_initializer)
            if d is not None:
                self._initialize_params(d)

    def _initialize_params(self, d):
        self.W.initialize((self.k, d))

    def __call__(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return straight_through(x, self.W)


class Encoder(chainer.Chain):
    def __init__(self, d=256):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, d, (4, 1), (2, 1), (1, 0))
            self.conv2 = L.Convolution2D(d, d, (4, 1), (2, 1), (1, 0))
            self.conv3 = L.Convolution2D(d, d, (4, 1), (2, 1), (1, 0))
            self.conv4 = L.Convolution2D(d, d, (4, 1), (2, 1), (1, 0))
            self.conv5 = L.Convolution2D(d, d, (4, 1), (2, 1), (1, 0))
            self.conv6 = L.Convolution2D(d, d, (4, 1), (2, 1), (1, 0))

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        z = self.conv6(h)
        return z


def gated(x, h=None):
    n_channel = x.shape[1]
    if h is not None:
        x = x + h
    return F.tanh(x[:, :n_channel // 2]) * F.sigmoid(x[:, n_channel // 2:])


class ResidualBlock(chainer.Chain):
    def __init__(self, dilation, n_channel1=32, n_channel2=16):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(n_channel2, n_channel1 * 2,
                                               ksize=(2, 1), pad=(dilation, 0),
                                               dilate=(dilation, 1))
            self.cond = L.DilatedConvolution2D(None, n_channel1 * 2,
                                               ksize=(2, 1), pad=(dilation, 0),
                                               dilate=(dilation, 1))
            self.proj = L.Convolution2D(n_channel1, n_channel2, 1)

        self.dilation = dilation
        self.n_channel2 = n_channel2

    def __call__(self, x, h=None):
        length = x.shape[2]

        # Dilated Conv
        x = self.conv(x)
        x = x[:, :, :length, :]

        if h is not None:
            h = self.cond(h)
            h = h[:, :, :length, :]

        # Gated activation units
        z = gated(x, h)

        # Projection
        z = self.proj(z)
        return z

    def initialize(self, n, cond=False):
        self.queue = chainer.Variable(
            self.xp.zeros((n, self.n_channel2, self.dilation + 1, 1),
                          dtype=self.xp.float32))
        self.conv.pad = (0, 0)
        if cond:
            self.cond_queue = chainer.Variable(
                self.xp.zeros((n, self.n_channel2, self.dilation + 1, 1),
                              dtype=self.xp.float32))
            self.cond.pad = (0, 0)
        else:
            self.cond = None

    def pop(self):
        return self.__call__(self.queue, self.cond_queue)

    def push(self, sample):
        self.queue = F.concat((self.queue[:, :, 1:, :], sample), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, n_filter,
                 n_channel1=32, n_channel2=16):
        super(ResidualNet, self).__init__()
        dilations = [
            n_filter ** i for j in range(n_loop) for i in range(n_layer)]
        for i, dilation in enumerate(dilations):
            self.add_link(ResidualBlock(dilation, n_channel1, n_channel2))

    def __call__(self, x, h=None):
        for i, func in enumerate(self.children()):
            a = x
            x = func(x, h)
            if i == 0:
                skip_connections = x
            else:
                skip_connections += x
            x = x + a
        return skip_connections

    def initialize(self, n, cond=False):
        for block in self.children():
            block.initialize(n, cond)

    def generate(self, x, h=None):
        sample = x
        for i, func in enumerate(self.children()):
            a = sample
            func.push(sample, h)
            sample = func.pop()
            if i == 0:
                skip_connections = sample
            else:
                skip_connections += sample
            sample = sample + a
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, n_filter, quantize=256,
                 n_channel1=32, n_channel2=16, n_channel3=512):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.caus = L.Convolution2D(
                quantize, n_channel2, (2, 1), pad=(1, 0))
            self.resb = ResidualNet(
                n_loop, n_layer, n_filter, n_channel1, n_channel2)
            self.proj1 = L.Convolution2D(n_channel2, n_channel3, 1)
            self.proj2 = L.Convolution2D(n_channel3, quantize, 1)
        self.n_layer = n_layer
        self.quantize = quantize
        self.n_channel2 = n_channel2
        self.n_channel3 = n_channel3

    def __call__(self, x, h=None):
        # Causal Conv
        length = x.shape[2]
        x = self.caus(x)
        x = x[:, :, :length, :]

        # Residual & Skip-conenection
        z = F.relu(self.resb(x, h))

        # Output
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        return y

    def initialize(self, n, cond=None):
        self.resb.initialize(n, cond)
        self.caus.pad = (0, 0)
        self.queue1 = chainer.Variable(
            self.xp.zeros((n, self.quantize, 2, 1), dtype=self.xp.float32))
        # self.queue1.data[:, self.quantize//2, :, :] = 1
        self.queue2 = chainer.Variable(
            self.xp.zeros((n, self.n_channel2, 1, 1), dtype=self.xp.float32))
        self.queue3 = chainer.Variable(
            self.xp.zeros((n, self.n_channel3, 1, 1), dtype=self.xp.float32))

    def generate(self, x, h=None):
        self.queue1 = F.concat((self.queue1[:, :, 1:, :], x), axis=2)
        x = self.caus(self.queue1)
        x = F.relu(self.resb.generate(x, h))
        self.queue2 = F.concat((self.queue2[:, :, 1:, :], x), axis=2)
        x = F.relu(self.proj1(self.queue2))
        self.queue3 = F.concat((self.queue3[:, :, 1:, :], x), axis=2)
        x = self.proj2(self.queue3)
        return x


class VAE(chainer.Chain):
    def __init__(self, k=512, beta=0.25):
        super(VAE, self).__init__()
        self.beta = beta
        with self.init_scope():
            self.enc = Encoder()
            self.vq = VQ(k)
            self.dec = WaveNet(3, 10, 2)

    def __call__(self, x, qt, t):
        # forward
        z = self.enc(x)
        e = self.vq(z)
        e_ = self.vq(z.data)
        y = self.dec(qt, F.unpooling_2d(e, (64, 1), cover_all=False))

        # calculate loss
        loss1 = F.softmax_cross_entropy(y, t)
        loss2 = F.mean((z.data - e_) ** 2)
        loss3 = self.beta * F.mean((z - e.data) ** 2)
        loss = loss1 + loss2 + loss3
        chainer.reporter.report(
            {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss': loss},
            self)
        return loss
