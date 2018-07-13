import numpy
import chainer
import chainer.functions as F
import chainer.links as L


class ResidualBlock(chainer.Chain):
    def __init__(self, filter_size, dilation,
                 residual_channels, dilated_channels, skip_channels,
                 condition_dim, dropout_zero_rate):
        super(ResidualBlock, self).__init__()
        with self.init_scope():
            self.conv = L.DilatedConvolution2D(
                residual_channels, dilated_channels,
                ksize=(filter_size, 1),
                pad=(dilation * (filter_size - 1), 0), dilate=(dilation, 1))
            self.condition_proj = L.Convolution2D(
                condition_dim, dilated_channels, 1)
            self.res = L.Convolution2D(
                dilated_channels // 2, residual_channels, 1)
            self.skip = L.Convolution2D(
                dilated_channels // 2, skip_channels, 1)

        self.filter_size = filter_size
        self.dilation = dilation
        self.residual_channels = residual_channels
        self.condition_dim = condition_dim
        self.dropout_zero_rate = dropout_zero_rate

    def __call__(self, x, condition):
        length = x.shape[2]

        # Dropout
        if self.dropout_zero_rate:
            h = F.dropout(x, ratio=self.dropout_zero_rate)
        else:
            h = x

        # Dilated conv
        h = self.conv(x)
        h = h[:, :, :length]

        # condition
        h += self.condition_proj(condition)

        # Gated activation units
        tanh_z, sig_z = F.split_axis(h, 2, axis=1)
        z = F.tanh(tanh_z) * F.sigmoid(sig_z)

        # Projection
        if x.shape[2] == z.shape[2]:
            residual = self.res(z) + x
        else:
            residual = self.res(z) + x[:, :, -1:]
        skip_conenection = self.skip(z)
        return residual, skip_conenection

    def initialize(self, n):
        self.queue = chainer.Variable(self.xp.zeros((
            n, self.residual_channels,
            self.dilation * (self.filter_size - 1) + 1, 1),
            dtype=self.xp.float32))
        self.conv.pad = (0, 0)
        self.condition_queue = chainer.Variable(
            self.xp.zeros((n, self.condition_dim, 1, 1),
                          dtype=self.xp.float32))

    def pop(self):
        return self(self.queue, self.condition_queue)

    def push(self, x, condition):
        self.queue = F.concat((self.queue[:, :, 1:], x), axis=2)
        self.condition_queue = F.concat(
            (self.condition_queue[:, :, 1:], condition), axis=2)


class ResidualNet(chainer.ChainList):
    def __init__(self, n_loop, n_layer, filter_size,
                 residual_channels, dilated_channels, skip_channels,
                 condition_dim, dropout_zero_rate):
        super(ResidualNet, self).__init__()
        dilations = [2 ** i for i in range(n_layer)] * n_loop
        for dilation in dilations:
            self.add_link(ResidualBlock(
                filter_size, dilation,
                residual_channels, dilated_channels, skip_channels,
                condition_dim, dropout_zero_rate))

    def __call__(self, x, condition):
        for i, func in enumerate(self.children()):
            x, skip = func(x, condition)
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections

    def initialize(self, n):
        for block in self.children():
            block.initialize(n)

    def generate(self, x, condition):
        for i, func in enumerate(self.children()):
            func.push(x, condition)
            x, skip = func.pop()
            if i == 0:
                skip_connections = skip
            else:
                skip_connections += skip
        return skip_connections


class WaveNet(chainer.Chain):
    def __init__(self, n_loop, n_layer, filter_size, input_dim,
                 residual_channels, dilated_channels, skip_channels,

                 # arguments for output
                 quantize, use_logistic, n_mixture, log_scale_min,

                 # arguments for conditioning
                 condition_dim,

                 # arguments for dropout
                 dropout_zero_rate):
        super(WaveNet, self).__init__()
        with self.init_scope():
            self.embed = L.Convolution2D(
                input_dim, residual_channels, (2, 1), pad=(1, 0))

            self.resnet = ResidualNet(
                n_loop, n_layer, filter_size,
                residual_channels, dilated_channels, skip_channels,
                condition_dim, dropout_zero_rate)

            self.proj1 = L.Convolution2D(skip_channels, skip_channels, 1)

            if use_logistic:
                output_dim = n_mixture
            else:
                output_dim = quantize
            self.proj2 = L.Convolution2D(skip_channels, output_dim, 1)

        self.input_dim = input_dim
        self.quantize = quantize
        self.skip_channels = skip_channels
        self.log_scale_min = log_scale_min

    def __call__(self, x, condition, generating=False):
        # Causal Conv
        length = x.shape[2]
        x = self.embed(x)
        x = x[:, :, :length, :]

        # Residual & Skip-conenection
        z = F.relu(self.resnet(x, condition))

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

    def calculate_logistic_loss(self, y, t):
        xp = chainer.cuda.get_array_module(t)
        if xp != numpy:
            xp.cuda.Device(t.device).use()
        nr_mix = y.shape[1] // 3

        logit_probs = y[:, :nr_mix]
        means = y[:, nr_mix:2 * nr_mix]
        log_scales = y[:, 2 * nr_mix:3 * nr_mix]
        log_scales = F.maximum(
            log_scales, self.scalar_to_tensor(log_scales, self.log_scale_min))

        t = F.broadcast_to(127.5 * t, means.shape)

        centered_t = t - means
        inv_std = F.exp(-log_scales)
        plus_in = inv_std * (centered_t + 127.5 / (self.quantize - 1))
        cdf_plus = F.sigmoid(plus_in)
        min_in = inv_std * (centered_t - 127.5 / (self.quantize - 1))
        cdf_min = F.sigmoid(min_in)

        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = -F.softplus(min_in)

        cdf_delta = cdf_plus - cdf_min

        # mid_in = inv_std * centered_t
        # log_pdf_mid = mid_in - log_scales - 2 * F.softplus(mid_in)

        log_probs = F.where(
            # condition
            t.array < self.scalar_to_tensor(t, 127.5 * -0.999),

            # true
            log_cdf_plus,

            # false
            F.where(
                # condition
                t.array > self.scalar_to_tensor(t, 127.5 * 0.999),

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

    def initialize(self, n):
        self.resnet.initialize(n)

        self.embed.pad = (0, 0)
        self.embed_queue = chainer.Variable(
            self.xp.zeros((n, self.input_dim, 2, 1), dtype=self.xp.float32))

        self.proj1_queue = chainer.Variable(self.xp.zeros(
            (n, self.skip_channels, 1, 1), dtype=self.xp.float32))

        self.proj2_queue3 = chainer.Variable(self.xp.zeros(
            (n, self.skip_channels, 1, 1), dtype=self.xp.float32))

    def generate(self, x, condition):
        self.embed_queue = F.concat((self.embed_queue[:, :, 1:], x), axis=2)
        x = self.embed(self.embed_queue)
        x = F.relu(self.resnet.generate(x, condition))

        self.proj1_queue = F.concat((self.proj1_queue[:, :, 1:], x), axis=2)
        x = F.relu(self.proj1(self.proj1_queue))

        self.proj2_queue3 = F.concat((self.proj2_queue3[:, :, 1:], x), axis=2)
        x = self.proj2(self.proj2_queue3)
        return x
