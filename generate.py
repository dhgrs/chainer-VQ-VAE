import argparse
import pathlib

import numpy
import librosa
import chainer

from WaveNet import WaveNet
from net import Encoder, ConditionEmbed
from utils import MuLaw
from utils import Preprocess
from utils import VQ
import params

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--output', '-o', help='output file')
parser.add_argument('--model', '-m', help='snapshot of trained model')
parser.add_argument('--speaker', '-s', default=None,
                    help='name of speaker. if this is None,'
                         'input speaker is used.')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu != [-1]:
    chainer.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
    chainer.global_config.autotune = True

# set data
path = args.input
if params.dataset_type == 'VCTK':
    n_speaker = len([
        speaker for speaker in pathlib.Path(params.root).glob('wav48/*/')])
elif params.dataset_type == 'ARCTIC':
    n_speaker = len([
        speaker for speaker in pathlib.Path(params.root).glob('*/')])
elif params.dataset_type == 'vs':
    n_speaker = len([
        speaker for speaker in pathlib.Path(params.root).glob('*/')])

# preprocess
n = 1  # batchsize; now suporrts only 1
preprocess = Preprocess(
    params.sr, params.top_db, params.input_dim, params.quantize, None,
    params.use_logistic, params.root, params.dataset_type)
inputs = preprocess(path)

x_enc, _, global_condition, _ = inputs
x_enc = numpy.expand_dims(x_enc, axis=0)
x_dec = numpy.zeros([n, params.input_dim, 1, 1], dtype=numpy.float32)

# make model
encoder = Encoder(params.d)
vq = VQ(params.k, params.d)
decoder = WaveNet(
    params.n_loop, params.n_layer, params.filter_size, params.input_dim,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.quantize, params.use_logistic, params.n_mixture,
    params.log_scale_min,
    params.local_condition_dim + params.global_condition_dim,
    params.dropout_zero_rate)
condition_embed = ConditionEmbed(
    n_speaker, params.global_condition_dim, params.local_condition_dim)

# load trained parameter
chainer.serializers.load_npz(
    args.model, encoder, 'updater/model:main/encoder/')
chainer.serializers.load_npz(args.model, vq, 'updater/model:main/vq/')
if params.ema_mu < 1:
    if params.use_ema:
        chainer.serializers.load_npz(
            args.model, decoder, 'updater/model:main/decoder/ema/')
    else:
        chainer.serializers.load_npz(
            args.model, decoder, 'updater/model:main/decoder/target/')
else:
    chainer.serializers.load_npz(
        args.model, decoder, 'updater/model:main/decoder/')
chainer.serializers.load_npz(
    args.model, condition_embed, 'updater/model:main/condition_embed/')

if args.gpu >= 0:
    use_gpu = True
    chainer.cuda.get_device_from_id(args.gpu).use()
else:
    use_gpu = False

if args.speaker in preprocess.speaker_dic:
    global_condition = numpy.asarray(
        [preprocess.speaker_dic[args.speaker]], dtype=numpy.int32)
else:
    global_condition = numpy.asarray([args.speaker], dtype=numpy.int32)

# forward
if use_gpu:
    x_enc = chainer.cuda.to_gpu(x_enc, device=args.gpu)
    x_dec = chainer.cuda.to_gpu(x_dec, device=args.gpu)
    global_condition = chainer.cuda.to_gpu(global_condition, device=args.gpu)
    encoder.to_gpu()
    vq.to_gpu()
    decoder.to_gpu()
    condition_embed.to_gpu()
x_dec = chainer.Variable(x_dec)
z = encoder(x_enc)
e = vq(z)
local_cond = e
condition = condition_embed(local_cond, global_condition)
decoder.initialize(n)
output = decoder.xp.zeros(condition.shape[2])

for i in range(len(output) - 1):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', params.apply_dropout):
            out = decoder.generate(x_dec, condition[:, :, i:i + 1]).array
    if params.use_logistic:
        nr_mix = out.shape[1] // 3

        logit_probs = out[:, :nr_mix]
        means = out[:, nr_mix:2 * nr_mix]
        log_scales = out[:, 2 * nr_mix:3 * nr_mix]
        log_scales = decoder.xp.maximum(log_scales, params.log_scale_min)

        # generate uniform
        rand = decoder.xp.random.uniform(0, 1, logit_probs.shape)

        # apply softmax
        prob = logit_probs - decoder.xp.log(-decoder.xp.log(rand))

        # sample
        argmax = prob.argmax(axis=1)
        means = means[:, argmax]
        log_scales = log_scales[:, argmax]

        # generate uniform
        rand = decoder.xp.random.uniform(0, 1, log_scales.shape)

        # convert into logistic
        rand = means + decoder.xp.exp(log_scales) * \
            (decoder.xp.log(rand) - decoder.xp.log(1 - rand))

        value = decoder.xp.squeeze(rand.astype(decoder.xp.float32))
        value /= 127.5
        x_dec.array[:] = value
    else:
        value = decoder.xp.random.choice(
            params.quantize,
            p=chainer.functions.softmax(out).array[0, :, 0, 0])
        zeros = decoder.xp.zeros_like(x_dec.array)
        zeros[:, value, :, :] = 1
        x_dec = chainer.Variable(zeros)
    output[i] = value

if use_gpu:
    output = chainer.cuda.to_cpu(output)
if params.use_logistic:
    wave = output
else:
    wave = MuLaw(params.quantize).itransform(output)
librosa.output.write_wav(args.output, wave, params.sr)
