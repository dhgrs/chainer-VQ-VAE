import chainer
import chainer.functions as F
import chainer.links as L
import numpy


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


class ResidualBlock(chainer.Chain):
    def __init__(self, dilation, residual_channels, dilated_channels,
                 skip_channels, embed_channels, d):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(
                residual_channels, dilated_channels * 2,
                ksize=(2, 1), pad=(dilation, 0), dilate=(dilation, 1))
            self.global_cond_embed = L.Linear(
                embed_channels, dilated_channels * 2)
            self.local_cond_conv = L.DilatedConvolution2D(
                d, dilated_channels * 2,
                ksize=(2, 1), pad=(dilation, 0), dilate=(dilation, 1))
            self.res = L.Convolution2D(dilated_channels, residual_channels, 1)
            self.skip = L.Convolution2D(dilated_channels, skip_channels, 1)

        self.dilation = dilation
        self.residual_channels = residual_channels
        self.dilated_channels = dilated_channels
        self.d = d

    def __call__(self, x, global_cond, local_cond):
        length = x.shape[2]

        # Dilated conv
        h = self.conv(x)
        h = h[:, :, :length]

        # global condition
        if global_cond is None:
            generating = True
            global_cond = self.global_cond
        else:
            generating = False
            global_cond = self.global_cond_embed(global_cond)
            global_cond = F.reshape(global_cond, global_cond.shape + (1, 1))
        global_cond = F.broadcast_to(global_cond, h.shape)

        # local condition
        local_cond = self.local_cond_conv(local_cond)
        local_cond = local_cond[:, :, :length]

        # Gated activation units
        z = h + global_cond + local_cond
        tanh_z, sig_z = F.split_axis(z, 2, axis=1)
        z = F.tanh(tanh_z) * F.sigmoid(sig_z)

        # Projection
        if generating:
            residual = self.res(z) + x[:, :, -1:]
        else:
            residual = self.res(z) + x
        skip_conenection = self.skip(z)
        return residual, skip_conenection

    def initialize(self, n, global_cond):
        self.queue = chainer.Variable(
            self.xp.zeros((n, self.residual_channels, self.dilation + 1, 1),
                          dtype=self.xp.float32))
        self.local_cond_queue = chainer.Variable(
            self.xp.zeros((n, self.d, self.dilation+1, 1),
                          dtype=self.xp.float32))
        self.conv.pad = (0, 0)
        self.local_cond_conv.pad = (0, 0)
        self.global_cond = self.global_cond_embed(global_cond)
        self.global_cond = F.reshape(
            self.global_cond, self.global_cond.shape + (1, 1))

    def pop(self):
        return self(self.queue, None, self.local_cond_queue)

    def push(self, x, local_cond):
        self.queue = F.concat((self.queue[:, :, 1:], x), axis=2)
        self.local_cond_queue = F.concat(
            (self.local_cond_queue[:, :, 1:], local_cond), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, n_filter, residual_channels,
                 dilated_channels, skip_channels, embed_channels, d):
        super(ResidualNet, self).__init__()
        dilations = [
            n_filter ** i for j in range(n_loop) for i in range(n_layer)]
        for i, dilation in enumerate(dilations):
            self.add_link(
                ResidualBlock(dilation, residual_channels, dilated_channels,
                              skip_channels, embed_channels, d))

    def __call__(self, x, global_cond, local_cond):
        for i, func in enumerate(self.children()):
            x, skip = func(x, global_cond, local_cond)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections

    def initialize(self, n, global_cond):
        for block in self.children():
            block.initialize(n, global_cond)

    def generate(self, x, local_cond):
        for i, func in enumerate(self.children()):
            func.push(x, local_cond)
            x, skip = func.pop()
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, n_filter, quantize, residual_channels,
                 dilated_channels, skip_channels, embed_channels, n_speaker, d):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.caus = L.Convolution2D(
                quantize, residual_channels, (2, 1), pad=(1, 0))
            self.resb = ResidualNet(
                n_loop, n_layer, n_filter, residual_channels, dilated_channels,
                skip_channels, embed_channels, d)
            self.proj1 = L.Convolution2D(skip_channels, skip_channels, 1)
            self.proj2 = L.Convolution2D(skip_channels, quantize, 1)
            self.embed = L.EmbedID(n_speaker, embed_channels)
        self.n_layer = n_layer
        self.quantize = quantize
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels

    def __call__(self, x, global_cond, local_cond):
        # Causal Conv
        length = x.shape[2]
        x = self.caus(x)
        x = x[:, :, :length, :]

        # Residual & Skip-conenection
        z = F.relu(self.resb(x, self.embed(global_cond), local_cond))

        # Output
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        return y

    def initialize(self, n, global_cond):
        self.resb.initialize(n, self.embed(global_cond))
        self.caus.pad = (0, 0)
        self.caus_queue = chainer.Variable(
            self.xp.zeros((n, self.quantize, 2, 1), dtype=self.xp.float32))
        # self.caus_queue.data[:, self.quantize//2, :, :] = 1
        self.proj1_queue = chainer.Variable(self.xp.zeros(
            (n, self.skip_channels, 1, 1), dtype=self.xp.float32))
        self.proj2_queue3 = chainer.Variable(self.xp.zeros(
            (n, self.skip_channels, 1, 1), dtype=self.xp.float32))

    def generate(self, x, local_cond):
        self.caus_queue = F.concat((self.caus_queue[:, :, 1:], x), axis=2)
        x = self.caus(self.caus_queue)

        x = F.relu(self.resb.generate(x, local_cond))

        self.proj1_queue = F.concat((self.proj1_queue[:, :, 1:], x), axis=2)
        x = F.relu(self.proj1(self.proj1_queue))

        self.proj2_queue3 = F.concat((self.proj2_queue3[:, :, 1:], x), axis=2)
        x = self.proj2(self.proj2_queue3)
        return x
