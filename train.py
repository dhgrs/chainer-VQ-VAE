import argparse
import datetime
import os
import shutil
import glob

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer
from chainer.training import extensions

from utils import Preprocess
from models import VAE
from updaters import VQVAE_StandardUpdater
from updaters import VQVAE_ParallelUpdater
import opt


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


# setup dataset iterator

if opt.dataset == 'VCTK':
    speakers = glob.glob(os.path.join(opt.root, 'wav48/*'))
    n_speaker = len(speakers)
    speaker_dic = {
        os.path.basename(speaker): i for i, speaker in enumerate(speakers)}
    files = glob.glob(os.path.join(opt.root, 'wav48/*/*.wav'))
# elif opt.dataset == 'ARCTIC':
#     files = glob.glob(os.path.join(opt.root, '*/wav/*.wav'))
#     valid_files = glob.glob(
#         os.path.join(opt.root, 'cmu_us_ksp_arctic/wav/*.wav'))

preprocess = Preprocess(
    opt.data_format, opt.sr, opt.mu, opt.top_db, opt.length, speaker_dic)

dataset = chainer.datasets.TransformDataset(files, preprocess)
train, valid = chainer.datasets.split_dataset(dataset, int(len(dataset) * 0.9))

# make directory of results
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, os.path.join(result, __file__))
shutil.copy('utils.py', os.path.join(result, 'utils.py'))
shutil.copy('models.py', os.path.join(result, 'models.py'))
shutil.copy('updaters.py', os.path.join(result, 'updaters.py'))
shutil.copy('opt.py', os.path.join(result, 'opt.py'))
shutil.copy('generate.py', os.path.join(result, 'generate.py'))

# Model
model = VAE(opt.d, opt.k, opt.n_loop, opt.n_layer, opt.n_filter, opt.mu,
            opt.n_channel1, opt.n_channel2, opt.n_channel3,
            opt.beta, n_speaker)

# Optimizer
optimizer = chainer.optimizers.Adam(opt.lr/len(args.gpus))
optimizer.setup(model)

# Iterator
if args.process * args.prefetch > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train, opt.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, opt.batchsize//len(args.gpus), repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)
else:
    train_iter = chainer.iterators.SerialIterator(train, opt.batchsize)
    valid_iter = chainer.iterators.SerialIterator(
        valid, opt.batchsize//len(args.gpus), repeat=False, shuffle=False)

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
trainer = chainer.training.Trainer(updater, opt.trigger, out=result)

# Extensions
trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpus[0]),
               trigger=opt.evaluate_interval)
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=opt.snapshot_interval)
trainer.extend(extensions.LogReport(trigger=opt.report_interval))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration',
     'main/loss1', 'main/loss2', 'main/loss3', 'validation/main/loss1',
     'validation/main/loss2', 'validation/main/loss3']),
    trigger=opt.report_interval)
trainer.extend(extensions.PlotReport(
    ['main/loss1', 'validation/main/loss1'],
    'iteration', file_name='loss1.png', trigger=opt.report_interval))
trainer.extend(extensions.PlotReport(
    ['main/loss2', 'validation/main/loss2'],
    'iteration', file_name='loss2.png', trigger=opt.report_interval))
trainer.extend(extensions.PlotReport(
    ['main/loss3', 'validation/main/loss3'],
    'iteration', file_name='loss3.png', trigger=opt.report_interval))
trainer.extend(extensions.ProgressBar(update_interval=100))

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# run
print('GPUs: {}'.format(*args.gpus))
print('# train: {}'.format(len(train)))
print('# valid: {}'.format(len(valid)))
print('# Minibatch-size: {}'.format(opt.batchsize))
print('# {}: {}'.format(opt.trigger[1], opt.trigger[0]))
print('')

trainer.run()
