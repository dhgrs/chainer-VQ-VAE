import chainer
import chainer.functions as F
import chainer.links as L


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
    def __init__(self, d):
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
    def __init__(self, dilation, n_channel1, n_channel2, conditional, d):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(
                n_channel2, n_channel1 * 2, ksize=(2, 1), pad=(dilation, 0),
                dilate=(dilation, 1))
            if conditional:
                self.cond = L.DilatedConvolution2D(
                    d, n_channel1 * 2, ksize=(2, 1), pad=(dilation, 0),
                    dilate=(dilation, 1))
            self.proj = L.Convolution2D(n_channel1, n_channel2 * 2, 1)

        self.dilation = dilation
        self.n_channel2 = n_channel2
        self.d = d
        self.conditional = conditional

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
        return z[:, :self.n_channel2], z[:, self.n_channel2:]

    def initialize(self, n):
        self.queue = chainer.Variable(
            self.xp.zeros((n, self.n_channel2, self.dilation + 1, 1),
                          dtype=self.xp.float32))
        self.conv.pad = (0, 0)
        if self.conditional:
            self.cond_queue = chainer.Variable(
                self.xp.zeros((n, self.d, self.dilation + 1, 1),
                              dtype=self.xp.float32))
            self.cond.pad = (0, 0)
        else:
            self.cond_queue = None

    def pop(self):
        return self.__call__(self.queue, self.cond_queue)

    def push(self, sample, h=None):
        self.queue = F.concat((self.queue[:, :, 1:, :], sample), axis=2)
        if h is not None:
            self.cond_queue = F.concat(
                (self.cond_queue[:, :, 1:, :], h), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, n_filter,
                 n_channel1, n_channel2, conditional, d):
        super(ResidualNet, self).__init__()
        dilations = [
            n_filter ** i for j in range(n_loop) for i in range(n_layer)]
        for i, dilation in enumerate(dilations):
            self.add_link(ResidualBlock(dilation, n_channel1, n_channel2,
                                        conditional, d))
        self.conditional = conditional

    def __call__(self, x, h=None):
        for i, func in enumerate(self.children()):
            a = x
            x, skip = func(x, h)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
            x = x + a
        return skip_connections

    def initialize(self, n):
        for block in self.children():
            block.initialize(n)

    def generate(self, x, h=None):
        sample = x
        for i, func in enumerate(self.children()):
            a = sample
            func.push(sample, h)
            sample, skip = func.pop()
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
            sample = sample + a
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, n_filter, quantize,
                 n_channel1, n_channel2, n_channel3,
                 conditional, d):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.caus = L.Convolution2D(
                None, n_channel2, (2, 1), pad=(1, 0))
            self.resb = ResidualNet(n_loop, n_layer, n_filter, n_channel1,
                                    n_channel2, conditional, d)
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

    def initialize(self, n):
        self.resb.initialize(n)
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
    def __init__(self, d=256, k=512, n_loop=3, n_layer=10, n_filter=2,
                 quantize='mixture', mu=256, n_channel1=32, n_channel2=16, n_channel3=512,
                 beta=0.25, conditional=True):
        super(VAE, self).__init__()
        self.beta = beta
        self.quantize = quantize
        self.mu = mu

        with self.init_scope():
            self.enc = Encoder(d)
            self.vq = VQ(k)
            self.dec = WaveNet(
                n_loop, n_layer, n_filter, mu,
                n_channel1, n_channel2, n_channel3,
                conditional, d)

    def __call__(self, x, t, qt=None):
        # forward
        z = self.enc(x)
        e = self.vq(z)
        e_ = self.vq(chainer.Variable(z.data))
        scale = t.shape[2] // e.shape[2]
        if self.quantize == 'mulaw':
            y_hat = self.dec(qt, F.unpooling_2d(e, (scale, 1), cover_all=False))
        elif self.quantize == 'mixture':
            y_hat = self.dec(x, F.unpooling_2d(e, (scale, 1), cover_all=False))

        # calculate loss
        if self.quantize == 'mulaw':
            loss1 = F.softmax_cross_entropy(y_hat, t)
        elif self.quantize == 'mixture':
            y_hat = y_hat[:, :30]
            logit_probs, means, log_scales = F.split_axis(y_hat, 3, 1)
            log_scales = F.relu(log_scales + 7) - 7
            y = F.broadcast_to(t, means.shape)

            centered_y = y - means
            inv_stdv = F.exp(-log_scales)
            plus_in = inv_stdv * (centered_y + 1 / (2 ** 16))
            cdf_plus = F.sigmoid(plus_in)
            min_in = inv_stdv * (centered_y - 1 / (2 ** 16))
            cdf_min = F.sigmoid(min_in)

            log_cdf_plus = plus_in - F.softplus(plus_in)
            log_one_minus_cdf_min = -F.softplus(min_in)

            cdf_delta = cdf_plus - cdf_min
            cdf_delta = F.relu(cdf_delta - 1e-12) + 1e-12

            y = F.broadcast_to(t, log_cdf_plus.shape).array
            log_probs = F.where(y < -0.999, log_cdf_plus, F.where(
                y > 0.999, log_one_minus_cdf_min, F.log(cdf_delta)))
            log_probs = log_probs + F.log_softmax(logit_probs)
            loss1 = -F.mean(log_probs)
        loss2 = F.mean((chainer.Variable(z.data) - e_) ** 2)
        loss3 = self.beta * F.mean((z - chainer.Variable(e.data)) ** 2)
        loss = loss1 + loss2 + loss3
        chainer.reporter.report(
            {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss': loss},
            self)
        return loss1, loss2, loss3

    def generate(self, x):
        # initialize and encode
        output = self.xp.zeros(x.shape[2])
        self.dec.initialize(1)
        with chainer.using_config('enable_backprop', False):
            z = self.enc(x)
            e = F.unpooling_2d(self.vq(z), (64, 1), cover_all=False)
        x = chainer.Variable(self.xp.zeros(
            self.mu, dtype=self.xp.float32).reshape((1, -1, 1, 1)))
        length = e.shape[2]

        # generate
        for i in range(length-1):
            with chainer.using_config('enable_backprop', False):
                out = self.dec.generate(x, e[:, :, i:i+1])
            zeros = self.xp.zeros_like(x.data)
            value = self.xp.random.choice(
                self.mu, p=F.softmax(out).data[0, :, 0, 0])
            output[i] = value
            zeros[:, value, :, :] = 1
            x = chainer.Variable(zeros)
        return output
