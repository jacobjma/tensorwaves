from abc import abstractmethod
from numbers import Number

import numpy as np
import tensorflow as tf

from tensorwaves.utils import batch_generator
import matplotlib.pyplot as plt
from collections import Iterable


class Scan(object):

    def __init__(self):
        self._detectors = {}
        self._num_positions = None

    def register_detector(self, detector):
        self._detectors[detector] = []

    def detect(self, wave):
        for detector, data in self._detectors.items():
            data.append(detector.detect(wave))

    def clear_detectors(self):
        self._detectors = {}

    def get_positions(self):
        raise NotImplementedError()

    @property
    def detectors(self):
        return self._detectors.keys()

    @property
    def num_positions(self):
        return self._num_positions

    def generate_positions(self, max_positions_per_batch):
        positions = self.get_positions()
        for start, stop in batch_generator(positions.shape[0].value, max_positions_per_batch):
            yield positions[start:start + stop]


class LineScan(Scan):

    def __init__(self, start, end, num_positions=None, sampling=None, endpoint=True):
        super().__init__()

        self._start = np.array(start, dtype=np.float32)
        self._end = np.array(end, dtype=np.float32)

        distance = np.sqrt(np.sum((self._end - self._start) ** 2))

        if (num_positions is None) & (sampling is not None):
            self._num_positions = np.ceil(distance / sampling).astype(np.int)
            if not endpoint:
                self._num_positions -= 1

        elif num_positions is not None:
            self._num_positions = num_positions

        else:
            raise TypeError('pass either argument: \'num_positions\' or \'sampling\'')

        if not endpoint:
            self._end -= (self._end - self._start) / self._num_positions

    def get_data(self, detector):
        return tf.concat(self._detectors[detector], axis=0)

    def show_data(self, detectors=None):
        if detectors is None:
            detectors = self.detectors

        elif not isinstance(detectors, Iterable):
            detectors = (detectors,)

        for detector in detectors:
            data = self.get_data(detector)
            plt.plot(data.numpy(),'.-')

    def get_positions(self):
        return tf.linspace(0., 1, self._num_positions)[:, None] * (self._end - self._start)[None] + self._start[None]


class GridScan(Scan):

    def __init__(self, start, end, num_positions=None, sampling=None, endpoint=False):

        def validate(value, dtype):
            if isinstance(value, np.ndarray):
                if len(value) != 2:
                    raise RuntimeError()
                return value.astype(dtype)

            elif isinstance(value, (list, tuple)):
                return map(dtype, value)

            elif isinstance(value, Number):
                return (dtype(value),) * 2

            else:
                raise RuntimeError()

        self._start = np.array(start, dtype=np.float32)
        self._end = np.array(end, dtype=np.float32)

        if (num_positions is None) & (sampling is not None):
            self._num_positions = np.ceil((self._start - self._end) / sampling).astype(np.int)
            if not endpoint:
                self._num_positions -= 1

        elif num_positions is not None:
            self._num_positions = validate(num_positions, dtype=np.int32)

        else:
            raise TypeError('pass either argument: \'num_positions\' or \'sampling\'')

        super().__init__()

    def get_positions(self):
        x = tf.linspace(self._start[0], self._end[0], self._num_positions[0])
        y = tf.linspace(self._start[1], self._end[1], self._num_positions[1])
        x, y = tf.meshgrid(x, y)
        return tf.stack((tf.reshape(x, (-1,)), tf.reshape(y, (-1,))), axis=1)
