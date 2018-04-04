import sys
import glob
import os
import numpy
import librosa
import chainer

from models import VAE
from utils import mu_law
from utils import Preprocess
import opt

# set data
if opt.dataset == 'VCTK':
    speakers = sorted(glob.glob(os.path.join(opt.root, 'wav48/*')))
    path = os.path.join(opt.root, 'wav48/p225/p225_001.wav')
elif opt.dataset == 'ARCTIC':
    speakers = sorted(glob.glob(os.path.join(opt.root, '*')))
    path = os.path.join(opt.root, 'cmu_us_bdl_arctic/wav/arctic_a0001.wav')
elif opt.dataset == 'vs':
    speakers = sorted(glob.glob(os.path.join(opt.root, '*')))
    path = os.path.join(opt.root, 'fujitou_normal/fujitou_normal_001.wav')

n_speaker = len(speakers)
speaker_dic = {
    os.path.basename(speaker): i for i, speaker in enumerate(speakers)}

# make model
model1 = VAE(
    opt.d, opt.k, opt.n_loop, opt.n_layer, opt.filter_size, opt.quantize,
    opt.residual_channels, opt.dilated_channels, opt.skip_channels,
    opt.use_logistic, opt.n_mixture, opt.log_scale_min, n_speaker,
    opt.embed_channels, opt.dropout_zero_rate, opt.ema_mu, opt.beta)
model2 = model1.copy()

# preprocess
n = 1
inputs = Preprocess(
    opt.data_format, opt.sr, opt.quantize, opt.top_db,
    opt.length, opt.dataset, speaker_dic, False)(path)

raw, one_hot, speaker, quantized = inputs
raw = numpy.expand_dims(raw, 0)
one_hot = numpy.expand_dims(one_hot, 0)

speaker = numpy.expand_dims(speaker, 0)
quantized = numpy.expand_dims(quantized, 0)

# forward
with chainer.using_config('enable_backprop', False):
    z = model1.enc(raw)
    e = model1.vq(z)
    global_cond = model1.dec.embed_global_cond(speaker)
    local_cond = model1.dec.upsample_local_cond(e)
model1.dec.initialize(n, global_cond)

print('check fast generation and naive generation')
for i in range(opt.sr):
    with chainer.using_config('enable_backprop', False):
        out1 = model1.dec.generate(
            one_hot[:, :, i:i+1], local_cond[:, :, i:i+1])
        out2 = model2.dec(
            one_hot[:, :, :i+1], global_cond, local_cond[:, :, :i+1],
            generating=True)
        print(
            '{}th sample, both of the values are same?:'.format(i),
            numpy.allclose(numpy.squeeze(out1.array),
                           numpy.squeeze(out2[:, :, -1:].array),
                           1e-3, 1e-5))
