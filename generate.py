import sys
import glob
import os
import argparse

import numpy
import librosa
import chainer

from models import VAE
from utils import mu_law
from utils import Preprocess
import opt

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--model', '-m', help='snapshot of trained model')
parser.add_argument('--speaker', '-s', default=None,
                    help='name of speaker. if this is None,'
                         'input speaker is used.')
args = parser.parse_args()

# set data
if opt.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(opt.root, 'wav48/*'))
elif opt.dataset == 'ARCTIC':
    speakers = glob.glob(os.path.join(opt.root, '*'))
path = args.input

n_speaker = len(speakers)
speaker_dic = {
    os.path.basename(speaker): i for i, speaker in enumerate(speakers)}

# make model
model = VAE(opt.d, opt.k, opt.n_loop, opt.n_layer, opt.n_filter, opt.mu,
            opt.residual_channels, opt.dilated_channels, opt.skip_channels,
            opt.beta, n_speaker)
chainer.serializers.load_npz(args.model, model, 'updater/model:main/')
# preprocess
n = 1
inputs = Preprocess(
    opt.data_format, opt.sr, opt.mu, opt.top_db,
    opt.length, opt.dataset, speaker_dic, False)(path)

raw, one_hot, speaker, quantized = inputs
raw = numpy.expand_dims(raw, 0)
one_hot = numpy.expand_dims(one_hot, 0)

print('from speaker', speaker)
if args.speaker is None:
    speaker = numpy.expand_dims(speaker, 0)
elif args.speaker in speaker_dic:
    speaker = numpy.asarray([speaker_dic[args.speaker]], dtype=numpy.int32)
else:
    speaker = numpy.asarray([args.speaker], dtype=numpy.int32)
print('to speaker', speaker[0])

quantized = numpy.expand_dims(quantized, 0)

# forward
output = model.generate(raw, speaker)
wave = mu_law(opt.mu).itransform(output)
numpy.save('result.npy', wave)
librosa.output.write_wav('result.wav', wave, opt.sr)
