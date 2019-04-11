from collections import Iterable
from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf

from tensorwaves.bases import Tensor
from tensorwaves.utils import batch_generator, ProgressBar


class Scan(object):

    def __init__(self, scanable=None, detectors=None, ):
        self._scanable = scanable
        if detectors is None:
            self._detectors = []
        else:

            if not isinstance(detectors, Iterable):
                self._detectors = [detectors]
            else:
                self._detectors = detectors

        self._data = None

    @property
    def scanable(self):
        return self._scanable

    @property
    def detectors(self):
        return self._detectors

    def get_positions(self):
        raise NotImplementedError('')

    @property
    def num_positions(self):
        raise NotImplementedError('')

    def generate_positions(self, max_batch, tracker=None):
        num_iter = (np.prod(self.num_positions) + max_batch - 1) // max_batch

        bar = ProgressBar(num_iter=num_iter, description='Scanning')

        # if tracker is not None:
        #     tracker.add_bar(bar)

        positions = self.get_positions()

        for i, (start, stop) in enumerate(batch_generator(positions.shape[0], max_batch)):
            bar.update(i)

            yield positions[start:start + stop]

        # if tracker is not None:
        #     del tracker._output[bar]

    def scan(self, max_batch=5, potential=None, tracker=None):

        if self.detectors:
            self._data = OrderedDict(zip(self.detectors, [[]] * len(self.detectors)))

        else:
            self._data = []

        for i, positions in enumerate(self.generate_positions(max_batch)):

            self.scanable._translate.positions = positions

            if potential is not None:
                tensor = self.scanable.multislice(potential)

            else:
                tensor = self.scanable.get_tensor()

            if self.detectors:
                for detector, detections in self._data.items():
                    detections.append(detector.detect(tensor))

            else:
                self._data.append(tensor)

        return self._data


class LineScan(Scan):

    def __init__(self, scanable, start, end, num_positions=None, detectors=None, sampling=None, endpoint=True):
        Scan.__init__(self, scanable=scanable, detectors=detectors)

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

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def num_positions(self):
        return self._num_positions

    def get_positions(self):
        return np.linspace(0., 1, self.num_positions)[:, None] * (self.end - self.start)[None] + self.start[None]

    def numpy(self, detector=None):
        if detector is None:
            detector = next(iter(self._data.keys()))

        return np.hstack(tf.convert_to_tensor(self._data[detector]).numpy())


class GridScan(Scan):

    def __init__(self, scanable=None, start=None, end=None, num_positions=None, detectors=None, sampling=None,
                 endpoint=False):

        Scan.__init__(self, scanable=scanable, detectors=detectors)

        self._shape = num_positions

        def validate(value, dtype):
            if isinstance(value, np.ndarray):
                if len(value) != 2:
                    raise RuntimeError()
                return value.astype(dtype)

            elif isinstance(value, (list, tuple)):
                return np.array(value, dtype=dtype)

            elif isinstance(value, Number):
                return np.array((dtype(value),) * 2)

            else:
                raise RuntimeError()

        if start is None:
            start = [0., 0.]

        if end is None:
            end = scanable.extent

        self._start = np.array(start, dtype=np.float32)
        self._end = np.array(end, dtype=np.float32)

        if (num_positions is None) & (sampling is not None):
            self._num_positions = np.ceil((np.abs(self._start - self._end)) / sampling).astype(np.int)
            if not endpoint:
                self._num_positions -= 1

        elif num_positions is not None:
            self._num_positions = validate(num_positions, dtype=np.int32)

        else:
            raise TypeError('pass either argument: \'num_positions\' or \'sampling\'')

        if not endpoint:
            self._end = self._end - np.abs(self._start - self._end) / np.array(self._num_positions)

    @property
    def num_positions(self):
        return np.prod(self._num_positions)

    def get_positions(self):
        x = np.linspace(self._start[0], self._end[0], self._num_positions[0])
        y = np.linspace(self._start[1], self._end[1], self._num_positions[1])
        y, x = np.meshgrid(y, x)

        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

    def image(self, detector=None):
        if detector is None:
            detector = next(iter(self._data.keys()))

        tensor = tf.reshape(tf.concat(self._data[detector], 0), self._num_positions)

        return Tensor(tensor[None], extent=self._end-self._start)

    #def show(self):

    def numpy(self, detector=None):
        if detector is None:
            detector = next(iter(self._data.keys()))

        return np.hstack(tf.convert_to_tensor(self._data[detector]).numpy()).reshape(self._num_positions)

    def split(self, n):
        positions = self.get_positions()

        N = positions.shape[0].value
        m = N // n
        scanners = []

        i = 0
        for i in range(n - 1):
            scanners.append(Scan(scanable=self._scanable, detectors=self.detectors,
                                 positions=positions[i * m:(i + 1) * m]))

        scanners.append(Scan(scanable=self._scanable, detectors=self.detectors,
                             positions=positions[(i + 1) * m:]))

        return scanners
