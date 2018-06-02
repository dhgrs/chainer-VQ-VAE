import random
import copy
import pathlib

import numpy
import librosa
import chainer
from chainer import configuration
from chainer import link


class MuLaw(object):
    def __init__(self, mu=256, int_type=numpy.int32, float_type=numpy.float32):
        self.mu = mu
        self.int_type = int_type
        self.float_type = float_type

    def transform(self, x):
        x = x.astype(self.float_type)
        y = numpy.sign(x) * numpy.log(1 + self.mu * numpy.abs(x)) / \
            numpy.log(1 + self.mu)
        y = numpy.digitize(y, 2 * numpy.arange(self.mu) / self.mu - 1) - 1
        return y.astype(self.int_type)

    def itransform(self, y):
        y = y.astype(self.float_type)
        y = 2 * y / self.mu - 1
        x = numpy.sign(y) / self.mu * ((self.mu) ** numpy.abs(y) - 1)
        return x.astype(self.float_type)


class Preprocess(object):
    def __init__(self, sr, res_type, top_db, input_dim,
                 quantize, length, use_logistic, root, dataset_type):
        self.sr = sr
        self.res_type = res_type
        self.top_db = top_db
        if input_dim == 1:
            self.mu_law_input = False
        else:
            self.mu_law_input = True
            self.mu_law = MuLaw(quantize)
            self.quantize = quantize
        if length is None:
            self.length = None
        else:
            self.length = length + 1
        self.use_logistic = use_logistic
        if self.mu_law_input or not self.use_logistic:
            self.mu_law = MuLaw(quantize)
            self.quantize = quantize
        self.speaker_dic = self.make_speaker_dic(root, dataset_type)

    def __call__(self, path):
        # load data(trim and normalize)
        raw, _ = librosa.load(path, self.sr, res_type=self.res_type)
        raw, _ = librosa.effects.trim(raw, self.top_db)
        raw /= numpy.abs(raw).max()
        raw = raw.astype(numpy.float32)

        # mu-law transform
        if self.mu_law_input or not self.use_logistic:
            quantized = self.mu_law.transform(raw)

        # padding/triming
        if self.length is not None:
            if len(raw) <= self.length:
                # padding
                pad = self.length - len(raw)
                raw = numpy.concatenate(
                    (raw, numpy.zeros(pad, dtype=numpy.float32)))
                if self.mu_law_input or not self.use_logistic:
                    quantized = numpy.concatenate(
                        (quantized, self.quantize // 2 * numpy.ones(pad)))
                    quantized = quantized.astype(numpy.int32)
            else:
                # triming
                start = random.randint(0, len(raw) - self.length - 1)
                raw = raw[start:start + self.length]
                if self.mu_law_input or not self.use_logistic:
                    quantized = quantized[start:start + self.length]

        # expand dimensions
        if self.mu_law_input:
            one_hot = numpy.identity(
                self.quantize, dtype=numpy.float32)[quantized]
            one_hot = numpy.expand_dims(one_hot.T, 2)
        raw = numpy.expand_dims(raw, 0)  # expand channel
        raw = numpy.expand_dims(raw, -1)  # expand height
        if not self.use_logistic:
            quantized = numpy.expand_dims(quantized, 1)

        if self.length is None:
            speaker_id = None  # `length is None` is a generating case
        else:
            speaker_id = numpy.array(
                self.speaker_dic[self.get_speaker(path)], dtype=numpy.int32)

        # return
        inputs = (raw,)
        if self.mu_law_input:
            inputs += (one_hot[:, :-1],)
        else:
            inputs += (raw[:, :-1],)
        inputs += (speaker_id,)
        if self.use_logistic:
            inputs += (raw[:, 1:],)
        else:
            inputs += (quantized[1:],)
        return inputs

    def make_speaker_dic(self, root, dataset_type):
        if dataset_type == 'VCTK':
            speakers = [
                str(speaker.name) for speaker in pathlib.Path(root).glob('wav48/*/')]
        elif dataset_type == 'ARCTIC':
            speakers = [
                str(speaker.name) for speaker in pathlib.Path(root).glob('*/')]
        elif dataset_type == 'vs':
            speakers = [
                str(speaker.name) for speaker in pathlib.Path(root).glob('*/')]
        speakers = sorted([speaker for speaker in speakers])
        speaker_dic = {speaker: i for i, speaker in enumerate(speakers)}
        return speaker_dic

    def get_speaker(self, path):
        speaker = pathlib.Path(path).parent.name
        return speaker


class ExponentialMovingAverage(link.Chain):

    def __init__(self, target, decay=0.999):
        super(ExponentialMovingAverage, self).__init__()
        self.decay = decay
        with self.init_scope():
            self.target = target
            self.ema = copy.deepcopy(target)

    def __call__(self, *args, **kwargs):
        if configuration.config.train:
            ys = self.target(*args, **kwargs)
            xp = chainer.cuda.get_array_module(ys)
            if xp != numpy:
                xp.cuda.Device(ys.array.device).use()
            for target_name, target_param in self.target.namedparams():
                for ema_name, ema_param in self.ema.namedparams():
                    if target_name == ema_name:
                        if not target_param.requires_grad \
                                or ema_param.array is None:
                            new_average = target_param.array
                        else:
                            new_average = self.decay * target_param.array + \
                                (1 - self.decay) * ema_param.array
                        ema_param.array = new_average
        else:
            ys = self.ema(*args, **kwargs)
        return ys


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