import chainer
import chainer.functions as F
import chainer.links as L
from utils import VQ
from utils import ExponentialMovingAverage


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


class ConditionEmbed(chainer.Chain):
    def __init__(self, n_global_cond, global_embed_dim, local_embed_dim,
                 upscale_factor=64):
        super(ConditionEmbed, self).__init__()
        with self.init_scope():
            self.local_embed1 = L.DilatedConvolution2D(
                None, local_embed_dim, (3, 1), pad=(1, 0), dilate=(1, 1))
            self.local_embed2 = L.DilatedConvolution2D(
                None, local_embed_dim, (3, 1), pad=(2, 0), dilate=(2, 1))
            self.local_embed3 = L.DilatedConvolution2D(
                None, local_embed_dim, (3, 1), pad=(4, 0), dilate=(4, 1))
            self.local_embed4 = L.DilatedConvolution2D(
                None, local_embed_dim, (3, 1), pad=(8, 0), dilate=(8, 1))
            self.local_embed5 = L.DilatedConvolution2D(
                None, local_embed_dim, (3, 1), pad=(16, 0), dilate=(16, 1))
            self.global_embed = L.EmbedID(n_global_cond, global_embed_dim)

        self.upscale_factor = upscale_factor

    def __call__(self, local_condition, global_condition):
        local_condition = F.relu(self.local_embed1(local_condition))
        local_condition = F.relu(self.local_embed2(local_condition))
        local_condition = F.relu(self.local_embed3(local_condition))
        local_condition = F.relu(self.local_embed4(local_condition))
        local_condition = F.relu(self.local_embed5(local_condition))
        local_condition = F.resize_images(
            local_condition, (self.upscale_factor * local_condition.shape[2], 1))

        global_condition = self.global_embed(global_condition)
        global_condition = global_condition.reshape(
            global_condition.shape + (1, 1))
        global_condition = F.resize_images(
            global_condition, (local_condition.shape[2], 1))

        condition = F.concat((local_condition, global_condition))
        return condition


class VAE(chainer.Chain):
    def __init__(self, encoder, decoder, condition_embed, d, k, beta,
                 loss_func):
        super(VAE, self).__init__()
        self.beta = beta
        self.loss_func = loss_func
        with self.init_scope():
            self.encoder = encoder
            self.vq = VQ(k, d)
            self.condition_embed = condition_embed
            self.decoder = decoder

    def __call__(self, x_enc, x_dec, global_condition, t):
        # forward
        z = self.encoder(x_enc)
        e = self.vq(z)
        e_ = self.vq(chainer.Variable(z.data))
        local_condition = e
        condition = self.condition_embed(local_condition, global_condition)
        y = self.decoder(x_dec, condition)

        # calculate loss
        loss1 = self.loss_func(y, t)
        loss2 = F.mean((chainer.Variable(z.data) - e_) ** 2)
        loss3 = self.beta * F.mean((z - chainer.Variable(e.data)) ** 2)
        loss = loss1 + loss2 + loss3
        chainer.reporter.report(
            {'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss': loss},
            self)
        return loss1, loss2, loss3
