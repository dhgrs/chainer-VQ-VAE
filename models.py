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
    def __init__(self, d=256):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, d, 4, 2, 1)
            self.conv2 = L.Convolution2D(d, d, 4, 2, 1)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        z = F.tanh(self.conv2(h))
        return z


class Decoder(chainer.Chain):
    def __init__(self, d=256):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.dconv1 = L.Deconvolution2D(d, d, 4, 2, 1)
            self.dconv2 = L.Deconvolution2D(d, 1, 4, 2, 1)

    def __call__(self, e, sigmoid=True):
        h = F.relu(self.dconv1(e))
        y = self.dconv2(h)
        if sigmoid:
            y = F.sigmoid(y)
        return y


class VAE(chainer.Chain):
    def __init__(self, k=512, beta=0.25, n_sample=5):
        super(VAE, self).__init__()
        self.beta = beta
        self.n_sample = n_sample
        with self.init_scope():
            self.enc = Encoder()
            self.vq = VQ(k)
            self.dec = Decoder()

    def __call__(self, x):
        # forward
        z = self.enc(x)
        e = self.vq(z)
        e_ = self.vq(z.data)
        y = self.dec(e, sigmoid=False)

        # calculate loss
        loss1 = F.mean(F.bernoulli_nll(x, y, reduce='no')) / self.n_sample
        loss2 = F.mean((z.data - e_) ** 2)
        loss3 = self.beta * F.mean((z - e.data) ** 2)
        loss = loss1 + loss2 + loss3
        chainer.reporter.report(
            {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss': loss},
            self)
        return loss

    def reconstract(self, x):
        return self.dec(self.vq(self.enc(x)))
