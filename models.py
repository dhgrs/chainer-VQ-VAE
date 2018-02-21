import chainer
import chainer.functions as F
import chainer.links as L
from modules import VQ
from modules import WaveNet


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


class VAE(chainer.Chain):
    def __init__(self, d, k, n_loop, n_layer, n_filter, quantize,
                 residual_channels, dilated_channels,
                 skip_channels, embed_channels, beta, n_speaker):
        super(VAE, self).__init__()
        self.beta = beta
        self.quantize = quantize
        with self.init_scope():
            self.enc = Encoder(d)
            self.vq = VQ(k)
            self.dec = WaveNet(
                n_loop, n_layer, n_filter, quantize, residual_channels,
                dilated_channels, skip_channels, embed_channels, n_speaker, d)

    def __call__(self, raw, one_hot, speaker, quantized):
        # forward
        z = self.enc(raw)
        e = self.vq(z)
        e_ = self.vq(chainer.Variable(z.data))
        scale = one_hot.shape[2] // e.shape[2]
        global_cond = speaker
        local_cond = F.unpooling_2d(e, (scale, 1), cover_all=False)
        y = self.dec(one_hot, global_cond, local_cond)

        # calculate loss
        loss1 = F.softmax_cross_entropy(y, quantized)
        loss2 = F.mean((chainer.Variable(z.data) - e_) ** 2)
        loss3 = self.beta * F.mean((z - chainer.Variable(e.data)) ** 2)
        loss = loss1 + loss2 + loss3
        chainer.reporter.report(
            {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss': loss},
            self)
        return loss1, loss2, loss3

    def generate(self, raw, speaker):
        # initialize and encode
        output = self.xp.zeros(raw.shape[2])
        with chainer.using_config('enable_backprop', False):
            z = self.enc(raw)
            e = self.vq(z)
            global_cond = speaker
            local_cond = F.unpooling_2d(e, (64, 1), cover_all=False)
        one_hot = chainer.Variable(self.xp.zeros(
            self.quantize, dtype=self.xp.float32).reshape((1, -1, 1, 1)))
        self.dec.initialize(1, global_cond)
        length = local_cond.shape[2]

        # generate
        for i in range(length-1):
            with chainer.using_config('enable_backprop', False):
                out = self.dec.generate(one_hot, local_cond[:, :, i:i+1])
            zeros = self.xp.zeros_like(one_hot.array)
            value = self.xp.random.choice(
                self.quantize, size=1, p=F.softmax(out).array[0, :, 0, 0])
            output[i] = value
            zeros[:, value, :, :] = 1
            one_hot = chainer.Variable(zeros)
        return output
