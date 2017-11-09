import argparse
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

from models import VAE
import opt


# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--GPU', '-G', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--process', '-p', type=int, default=1,
                    help='Number of parallel processes')
parser.add_argument('--prefetch', '-f', type=int, default=1,
                    help='Number of prefetch samples')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
args = parser.parse_args()


# setup dataset iterator
train, valid = chainer.datasets.get_mnist(False, 3)

# make directory of results
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, result + '/' + __file__)
shutil.copy('models.py', result + '/' + 'models.py')
shutil.copy('opt.py', result + '/' + 'opt.py')

# Model
gpu = max(args.gpu, args.GPU)
if args.gpu >= 0 and args.GPU >= 0:
    chainer.cuda.get_device_from_id(gpu).use()
model = VAE()

# Optimizer
optimizer = chainer.optimizers.Adam(opt.lr)
optimizer.setup(model)

# Iterator
if args.process * args.prefetch > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train, opt.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, opt.batchsize, repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)
else:
    train_iter = chainer.iterators.SerialIterator(train, opt.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, opt.batchsize,
                                                  repeat=False, shuffle=False)

# Updater
if args.gpu >= 0 and args.GPU >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    updater = chainer.training.ParallelUpdater(
        train_iter, optimizer,
        devices={'main': gpu, 'second': min(args.gpu, args.GPU)})
elif args.gpu >= 0 or args.GPU >= 0:
    chainer.cuda.get_device_from_id(args.gpu).use()
    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=gpu)
else:
    updater = chainer.training.StandardUpdater(train_iter, optimizer)

# Trainer
trainer = chainer.training.Trainer(updater, opt.trigger, out=result)

# Extensions
trainer.extend(extensions.Evaluator(valid_iter, model, device=gpu),
               trigger=opt.report_interval)
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot_object(model, 'model{.updater.iteration}'),
               trigger=opt.report_interval)
trainer.extend(extensions.LogReport(trigger=(5, 'iteration')))
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss1', 'main/loss2', 'main/loss3',
     'validation/loss1', 'validation/loss2', 'validation/loss3']),
    trigger=(5, 'iteration'))
trainer.extend(extensions.PlotReport(
    ['main/loss1', 'main/loss2', 'main/loss3',
     'validation/loss1', 'validation/loss2', 'validation/loss3'],
    'iteration', file_name='loss.png', trigger=(5, 'iteration')))
trainer.extend(extensions.ProgressBar(update_interval=5))

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# run
print('GPU1: {}'.format(args.gpu))
print('GPU2: {}'.format(args.GPU))
print('# Minibatch-size: {}'.format(opt.batchsize))
print('# {}: {}'.format(opt.trigger[1], opt.trigger[0]))
print('')

trainer.run()
