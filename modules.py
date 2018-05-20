import chainer
import chainer.functions as F
import chainer.links as L


class NonCausalDilatedConvNet(chainer.ChainList):
    def __init__(self, channels):
        super(NonCausalDilatedConvNet, self).__init__()
        for layer, channel in enumerate(channels):
            dilation = 2 ** layer
            self.add_link(L.DilatedConvolution2D(
                None, channel, (3, 1), pad=(dilation, 0),
                dilate=(dilation, 1)))

    def __call__(self, x):
        for link in self.children():
            x = F.relu(link(x))
        return x


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
    def __init__(self, filter_size, dilation, residual_channels,
                 dilated_channels, skip_channels, global_conditioned,
                 local_conditioned, embed_dim, local_condition_dim,
                 dropout_zero_rate):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(
                residual_channels, dilated_channels,
                ksize=(filter_size, 1),
                pad=(dilation * (filter_size - 1), 0), dilate=(dilation, 1))
            if global_conditioned:
                self.global_cond_proj = L.Convolution2D(
                    embed_dim, dilated_channels, 1)
            if local_conditioned:
                self.local_cond_proj = L.Convolution2D(
                    local_condition_dim, dilated_channels, 1)
            self.res = L.Convolution2D(
                dilated_channels // 2, residual_channels, 1)
            self.skip = L.Convolution2D(
                dilated_channels // 2, skip_channels, 1)

        self.filter_size = filter_size
        self.dilation = dilation
        self.residual_channels = residual_channels
        self.dilated_channels = dilated_channels
        self.global_conditioned = global_conditioned
        self.local_conditioned = local_conditioned
        self.local_condition_dim = local_condition_dim
        self.dropout_zero_rate = dropout_zero_rate

    def __call__(self, x, global_cond, local_cond):
        length = x.shape[2]

        # Dilated conv
        h = self.conv(x)
        h = h[:, :, :length]

        # global condition
        if self.global_conditioned:
            if global_cond is None:
                global_cond = self.global_cond
            else:
                global_cond = self.global_cond_proj(global_cond)
                global_cond = F.broadcast_to(global_cond, h.shape)
            h += global_cond

        # local condition
        if self.local_conditioned:
            if local_cond is not None:
                local_cond = self.local_cond_proj(local_cond)
                h += local_cond
            else:
                print('local condition is not feed')

        # Gated activation units
        if self.dropout_zero_rate:
            h = F.dropout(h, ratio=self.dropout_zero_rate)
        tanh_z, sig_z = F.split_axis(h, 2, axis=1)
        z = F.tanh(tanh_z) * F.sigmoid(sig_z)

        # Projection
        if x.shape[2] == z.shape[2]:
            residual = self.res(z) + x
        else:
            residual = self.res(z) + x[:, :, -1:]
        skip_conenection = self.skip(z)
        return residual, skip_conenection

    def initialize(self, n, global_cond):
        self.queue = chainer.Variable(self.xp.zeros((
            n, self.residual_channels,
            self.dilation * (self.filter_size - 1) + 1, 1),
            dtype=self.xp.float32))
        self.conv.pad = (0, 0)
        if self.local_conditioned:
            self.local_cond_queue = chainer.Variable(
                self.xp.zeros(
                    (n, self.local_condition_dim, 1, 1),
                    dtype=self.xp.float32))
        else:
            self.local_cond_queue = None
        if self.global_conditioned:
            self.global_cond = self.global_cond_proj(global_cond)
        else:
            self.global_cond = None

    def pop(self):
        return self(self.queue, None, self.local_cond_queue)

    def push(self, x, local_cond):
        self.queue = F.concat((self.queue[:, :, 1:], x), axis=2)
        if self.local_conditioned:
            self.local_cond_queue = F.concat(
                (self.local_cond_queue[:, :, 1:], local_cond), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, filter_size, residual_channels,
                 dilated_channels, skip_channels, global_conditioned,
                 local_conditioned, embed_dim, local_condition_dim,
                 dropout_zero_rate):
        super(ResidualNet, self).__init__()
        dilations = [
            2 ** i for j in range(n_loop) for i in range(n_layer)]
        for i, dilation in enumerate(dilations):
            self.add_link(ResidualBlock(
                filter_size, dilation, residual_channels, dilated_channels,
                skip_channels, global_conditioned, local_conditioned,
                embed_dim, local_condition_dim, dropout_zero_rate))

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
            # print(i, x.shape)
            x, skip = func.pop()
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, filter_size, quantize,
                 residual_channels, dilated_channels, skip_channels,
                 use_logistic, global_conditioned, local_conditioned,
                 # arguments for mixture of logistics
                 n_mixture, log_scale_min,
                 # arguments for global condition
                 n_speaker, embed_dim,
                 # arguments for local conditon
                 local_condition_dim, upsample_factor, use_deconv,
                 # arguments for dropout
                 dropout_zero_rate):
        super(WaveNet, self).__init__()
        with self.init_scope():
            if local_conditioned:
                self.local_embed = NonCausalDilatedConvNet(
                    [local_condition_dim] * 5)
                if use_deconv:
                    self.upsample = L.Deconvolution2D(
                        local_condition_dim, local_condition_dim,
                        (upsample_factor, 1), (upsample_factor, 1))

            if global_conditioned:
                self.embed = L.EmbedID(n_speaker, embed_dim)

            if use_logistic:
                self.caus = L.Convolution2D(
                    1, residual_channels, (2, 1), pad=(1, 0))
            else:
                self.caus = L.Convolution2D(
                    quantize, residual_channels, (2, 1), pad=(1, 0))

            self.resb = ResidualNet(
                n_loop, n_layer, filter_size,
                residual_channels, dilated_channels, skip_channels,
                global_conditioned, local_conditioned,
                embed_dim, local_condition_dim, dropout_zero_rate)

            self.proj1 = L.Convolution2D(skip_channels, skip_channels, 1)

            if use_logistic:
                self.proj2 = L.Convolution2D(skip_channels, n_mixture, 1)
            else:
                self.proj2 = L.Convolution2D(skip_channels, quantize, 1)

        self.n_layer = n_layer
        self.quantize = quantize
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.global_conditioned = global_conditioned
        self.local_conditioned = local_conditioned
        self.upsample_factor = upsample_factor
        self.use_deconv = use_deconv
        self.use_logistic = use_logistic
        self.log_scale_min = log_scale_min

    def __call__(self, x, global_cond=None, local_cond=None,
                 generating=False):
        if self.local_conditioned:
            if not generating:
                local_cond = self.upsample_local_cond(local_cond)
        else:
            local_cond = None

        if self.global_conditioned:
            if not generating:
                global_cond = self.embed_global_cond(global_cond)
        else:
            global_cond = None

        # Causal Conv
        length = x.shape[2]
        x = self.caus(x)
        x = x[:, :, :length, :]

        # Residual & Skip-conenection
        z = F.relu(self.resb(x, global_cond, local_cond))

        # Output
        z = F.relu(self.proj1(z))
        y = self.proj2(z)
        return y

    def scalar_to_tensor(self, shapeortensor, scalar):
        if hasattr(shapeortensor, 'shape'):
            shape = shapeortensor.shape
        else:
            shape = shapeortensor
        return self.xp.full(shape, scalar, dtype=self.xp.float32)

    def calculate_logistic_loss(self, y_hat, y):
        nr_mix = y_hat.shape[1] // 3

        logit_probs = y_hat[:, :nr_mix]
        means = y_hat[:, nr_mix:2 * nr_mix]
        log_scales = y_hat[:, 2 * nr_mix:3 * nr_mix]
        log_scales = F.maximum(
            log_scales, self.scalar_to_tensor(log_scales, self.log_scale_min))

        y = F.broadcast_to(y, means.shape)

        centered_y = y - means
        inv_stdv = F.exp(-log_scales)
        plus_in = inv_stdv * (centered_y + 1 / (self.quantize - 1))
        cdf_plus = F.sigmoid(plus_in)
        min_in = inv_stdv * (centered_y - 1 / (self.quantize - 1))
        cdf_min = F.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)

        cdf_delta = cdf_plus - cdf_min

        # mid_in = inv_stdv * centered_y
        # log_pdf_mid = mid_in - log_scales - 2 * F.softplus(mid_in)

        log_probs = F.where(
            # condition
            y.array < self.scalar_to_tensor(y, -0.999),

            # true
            log_cdf_plus,

            # false
            F.where(
                # condition
                y.array > self.scalar_to_tensor(y, 0.999),

                # true
                log_one_minus_cdf_min,

                # false
                F.log(F.maximum(
                    cdf_delta, self.scalar_to_tensor(cdf_delta, 1e-12)))
                # F.where(
                #     # condition
                #     cdf_delta.array > self.scalar_to_tensor(cdf_delta, 1e-5),

                #     # true
                #     F.log(F.maximum(
                #         cdf_delta, self.scalar_to_tensor(cdf_delta, 1e-12))),

                #     # false
                #     log_pdf_mid - self.xp.log((self.quantize - 1) / 2))
                ))

        log_probs = log_probs + F.log_softmax(logit_probs)
        loss = -F.mean(F.logsumexp(log_probs, axis=1))
        return loss

    def initialize(self, n, global_cond):
        self.resb.initialize(n, global_cond)
        self.caus.pad = (0, 0)
        if self.use_logistic:
            self.caus_queue = chainer.Variable(
                self.xp.zeros((n, 1, 2, 1), dtype=self.xp.float32))
        else:
            self.caus_queue = chainer.Variable(
                self.xp.zeros((n, self.quantize, 2, 1), dtype=self.xp.float32))
        # self.caus_queue.data[:, self.quantize//2, :, :] = 1
        self.proj1_queue = chainer.Variable(self.xp.zeros(
            (n, self.skip_channels, 1, 1), dtype=self.xp.float32))
        self.proj2_queue3 = chainer.Variable(self.xp.zeros(
            (n, self.skip_channels, 1, 1), dtype=self.xp.float32))

    def upsample_local_cond(self, local_cond):
        if self.use_deconv:
            local_cond = self.upsample(local_cond)
        else:
            local_cond = F.resize_images(
                local_cond, (local_cond.shape[2] * self.upsample_factor, 1))
        local_cond = self.local_embed(local_cond)
        return local_cond

    def embed_global_cond(self, global_cond):
        global_cond = self.embed(global_cond)
        global_cond = F.reshape(global_cond, global_cond.shape + (1, 1))
        return global_cond

    def generate(self, x, local_cond):
        self.caus_queue = F.concat((self.caus_queue[:, :, 1:], x), axis=2)
        x = self.caus(self.caus_queue)
        x = F.relu(self.resb.generate(x, local_cond))

        self.proj1_queue = F.concat((self.proj1_queue[:, :, 1:], x), axis=2)
        x = F.relu(self.proj1(self.proj1_queue))

        self.proj2_queue3 = F.concat((self.proj2_queue3[:, :, 1:], x), axis=2)
        x = self.proj2(self.proj2_queue3)
        return x
