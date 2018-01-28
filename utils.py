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
    def __init__(self, data_format, sr, mu, top_db, length, speaker_dic,
                 random=True):
        self.data_format = data_format
        self.sr = sr
        self.mu = mu
        self.mu_law = mu_law(mu)
        self.top_db = top_db
        self.length = length + 1
        self.speaker_dic = speaker_dic
        self.random = random

    def __call__(self, path):
        # load data
        raw = self.read_file(path)
        raw, _ = librosa.effects.trim(raw, self.top_db)
        raw /= np.abs(raw).max()

        qt = self.mu_law.transform(raw)

        if len(raw) <= self.length:
            # padding
            pad = self.length-len(raw)
            raw = np.concatenate(
                (raw, np.zeros(pad, dtype=np.float32)))
            qt = np.concatenate(
                (qt, self.mu // 2 * np.ones(pad, dtype=np.int32)))
        else:
            # triming
            if self.random:
                start = random.randint(0, len(raw) - self.length-1)
                raw = raw[start:start + self.length]
                qt = qt[start:start + self.length]
            else:
                raw = raw[:self.length]
                qt = qt[:self.length]

        # expand dimension
        raw = raw.reshape((1, -1, 1))
        y = np.identity(self.mu)[qt].astype(np.float32)
        y = np.expand_dims(y.T, 2)
        t = np.expand_dims(qt.astype(np.int32), 1)

        # get speaker-id
        speaker = self.speaker_dic[os.path.basename(os.path.dirname(path))]

        return raw[:, :-1, :], y[:, :-1, :], np.int32(speaker), t[1:, :]

    def read_file(self, path):
        x, sr = librosa.core.load(path, self.sr, res_type='kaiser_fast')
        return x
