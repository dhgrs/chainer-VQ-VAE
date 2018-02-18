# coding: UTF-8
import random
import os

import numpy as np
import librosa


class mu_law(object):
    def __init__(self, mu=256, int_type=np.uint8, float_type=np.float32):
        self.mu = mu
        self.int_type = int_type
        self.float_type = float_type

    def transform(self, x):
        x = x.astype(self.float_type)
        y = np.sign(x) * np.log(1 + self.mu * np.abs(x)) / np.log(1 + self.mu)
        y = np.digitize(y, 2 * np.arange(self.mu) / self.mu - 1) - 1
        return y.astype(self.int_type)

    def itransform(self, y):
        y = y.astype(self.float_type)
        y = 2 * y / self.mu - 1
        x = np.sign(y) / self.mu * ((self.mu) ** np.abs(y) - 1)
        return x.astype(self.float_type)


class Preprocess(object):
    def __init__(self, data_format, sr, mu, top_db, length, dataset,
                 speaker_dic, random=True):
        self.data_format = data_format
        self.sr = sr
        self.mu = mu
        self.mu_law = mu_law(mu)
        self.top_db = top_db
        if length is None:
            self.length = None
        else:
            self.length = length + 1
        self.dataset = dataset
        self.speaker_dic = speaker_dic
        self.random = random

    def __call__(self, path):
        # load data
        raw = self.read_file(path)
        raw, _ = librosa.effects.trim(raw, self.top_db)
        raw /= np.abs(raw).max()

        quantized = self.mu_law.transform(raw)

        if self.length is not None:
            if len(raw) <= self.length:
                # padding
                pad = self.length-len(raw)
                raw = np.concatenate(
                    (raw, np.zeros(pad, dtype=np.float32)))
                quantized = np.concatenate(
                    (quantized, self.mu // 2 * np.ones(pad, dtype=np.int32)))
            else:
                # triming
                if self.random:
                    start = random.randint(0, len(raw) - self.length-1)
                    raw = raw[start:start + self.length]
                    quantized = quantized[start:start + self.length]
                else:
                    raw = raw[:self.length]
                    quantized = quantized[:self.length]

        # expand dimension
        raw = raw.reshape((1, -1, 1))
        one_hot = np.identity(self.mu)[quantized].astype(np.float32)
        one_hot = np.expand_dims(one_hot.T, 2)
        quantized = np.expand_dims(quantized.astype(np.int32), 1)

        # get speaker-id
        if self.dataset == 'VCTK':
            try:
                speaker = self.speaker_dic[
                    os.path.basename(os.path.dirname(path))]
            except:
                speaker = None
        elif self.dataset == 'ARCTIC':
            try:
                speaker = self.speaker_dic[
                    os.path.basename(os.path.dirname(os.path.dirname(path)))]
            except:
                speaker = None
        speaker = np.int32(speaker)
        return raw[:, :-1, :], one_hot[:, :-1, :], speaker, quantized[1:, :]

    def read_file(self, path):
        x, sr = librosa.core.load(path, self.sr, res_type='kaiser_fast')
        return x
