import argparse
import pathlib
import datetime
import os
import shutil

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer
from chainer.training import extensions

from utils import Preprocess
from utils import ExponentialMovingAverage
from WaveNet import WaveNet
from net import Encoder, ConditionEmbed, VAE
from updaters import VQVAE_StandardUpdater, VQVAE_ParallelUpdater
import params


# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', '-g', type=int, default=[-1], nargs='+',
                    help='GPU IDs (negative value indicates CPU)')
parser.add_argument('--process', '-p', type=int, default=1,
                    help='Number of parallel processes')
parser.add_argument('--prefetch', '-f', type=int, default=1,
                    help='Number of prefetch samples')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
args = parser.parse_args()
if args.gpus != [-1]:
    chainer.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
    chainer.global_config.autotune = True

# get paths
if params.dataset_type == 'VCTK':
    files = sorted([
        str(path) for path in pathlib.Path(params.root).glob('wav48/*/*.wav')])
    n_speaker = len([
        speaker for speaker in pathlib.Path(params.root).glob('wav48/*/')])
elif params.dataset_type == 'ARCTIC':
    files = sorted([
        str(path) for path in pathlib.Path(params.root).glob('*/wav/*.wav')])
    n_speaker = len([
        speaker for speaker in pathlib.Path(params.root).glob('*/')])
elif params.dataset_type == 'vs':
    files = sorted([
        str(path) for path in pathlib.Path(params.root).glob('*/*.wav')])
    n_speaker = len([
        speaker for speaker in pathlib.Path(params.root).glob('*/')])

preprocess = Preprocess(
    params.sr, params.top_db, params.input_dim, params.quantize, params.length,
    params.use_logistic, params.root, params.dataset_type)

dataset = chainer.datasets.TransformDataset(files, preprocess)
train, valid = chainer.datasets.split_dataset_random(
    dataset, int(len(dataset) * 0.9), params.split_seed)

# make directory of results
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, os.path.join(result, __file__))
shutil.copy('utils.py', os.path.join(result, 'utils.py'))
shutil.copy('params.py', os.path.join(result, 'params.py'))
shutil.copy('generate.py', os.path.join(result, 'generate.py'))
shutil.copy('net.py', os.path.join(result, 'net.py'))
shutil.copy('updaters.py', os.path.join(result, 'updaters.py'))
shutil.copytree('WaveNet', os.path.join(result, 'WaveNet'))

# Model
encoder = Encoder(params.d)
wavenet = WaveNet(
    params.n_loop, params.n_layer, params.filter_size, params.input_dim,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.quantize, params.use_logistic, params.n_mixture,
    params.log_scale_min,
    params.local_condition_dim + params.global_condition_dim,
    params.dropout_zero_rate)
condition_embed = ConditionEmbed(
    n_speaker, params.global_condition_dim, params.local_condition_dim)

if params.ema_mu < 1:
    decoder = ExponentialMovingAverage(wavenet, params.ema_mu)
else:
    decoder = wavenet

if params.use_logistic:
    loss_fun = wavenet.calculate_logistic_loss
else:
    loss_fun = chainer.functions.softmax_cross_entropy
model = VAE(
    encoder, decoder, condition_embed, params.d, params.k, params.beta,
    loss_fun)

# Optimizer
optimizer = chainer.optimizers.Adam(params.lr/len(args.gpus))
optimizer.setup(model)

# Iterator
if args.process * args.prefetch > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train, params.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, params.batchsize//len(args.gpus), repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)
else:
    train_iter = chainer.iterators.SerialIterator(train, params.batchsize)
    valid_iter = chainer.iterators.SerialIterator(
        valid, params.batchsize//len(args.gpus), repeat=False, shuffle=False)

# Updater
if args.gpus == [-1]:
    updater = VQVAE_StandardUpdater(train_iter, optimizer)
else:
    chainer.cuda.get_device_from_id(args.gpus[0]).use()
    names = ['main'] + list(range(len(args.gpus)-1))
    devices = {str(name): gpu for name, gpu in zip(names, args.gpus)}
    updater = VQVAE_ParallelUpdater(
        train_iter, optimizer, devices=devices)

# Trainer
trainer = chainer.training.Trainer(updater, params.trigger, out=result)

# Extensions
trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpus[0]),
               trigger=params.evaluate_interval)
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=params.snapshot_interval)
trainer.extend(extensions.LogReport(trigger=params.report_interval))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration',
     'main/loss1', 'main/loss2', 'main/loss3', 'validation/main/loss1',
     'validation/main/loss2', 'validation/main/loss3']),
    trigger=params.report_interval)
trainer.extend(extensions.PlotReport(
    ['main/loss1', 'validation/main/loss1'],
    'iteration', file_name='loss1.png', trigger=params.report_interval))
trainer.extend(extensions.PlotReport(
    ['main/loss2', 'validation/main/loss2'],
    'iteration', file_name='loss2.png', trigger=params.report_interval))
trainer.extend(extensions.PlotReport(
    ['main/loss3', 'validation/main/loss3'],
    'iteration', file_name='loss3.png', trigger=params.report_interval))
trainer.extend(extensions.ProgressBar(update_interval=1))

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# run
print('GPUs: {}'.format(*args.gpus))
print('# train: {}'.format(len(train)))
print('# valid: {}'.format(len(valid)))
print('# Minibatch-size: {}'.format(params.batchsize))
print('# {}: {}'.format(params.trigger[1], params.trigger[0]))
print('')

trainer.run()
