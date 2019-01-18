from collections import Iterable, OrderedDict
from numbers import Number

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorwaves.bases import HasData
from tensorwaves.detect import Image
from tensorwaves.utils import batch_generator, ProgressBar


class Scan(HasData):

    def __init__(self, scanable, detectors, save_data=True):
        self._scanable = scanable

        if not isinstance(detectors, Iterable):
            detectors = (detectors,)

        for detector in detectors:
            detector.register_observer(self)

        self._detectors = detectors

        self._num_positions = None

        HasData.__init__(self, save_data=save_data)

    def get_positions(self):
        raise NotImplementedError()

    @property
    def detectors(self):
        return self._detectors

    @property
    def num_positions(self):
        return self._num_positions

    def generate_positions(self, max_batch):
        positions = self.get_positions()
        for start, stop in batch_generator(positions.shape[0].value, max_batch):
            yield positions[start:start + stop]

    def read_detector(self, detector=None):
        data = self.get_data()

        if detector is None:
            detector = next(iter(data))

        return tf.reshape(tf.concat(data[detector], axis=0), tf.reshape(self.num_positions, (-1,)))

    def scan(self, max_batch=1, potential=None, tracker=None):
        self._data = self._calculate_data(max_batch=max_batch, potential=potential, tracker=tracker)
        self.up_to_date = True

        # return self.get_data()

    def _calculate_data(self, max_batch=1, potential=None, tracker=None):

        data = OrderedDict(zip(self.detectors, [[]] * len(self.detectors)))

        num_iter = (np.prod(self.num_positions) + max_batch - 1) // max_batch

        bar = ProgressBar(num_iter=num_iter, description='Scanning')

        if tracker is not None:
            tracker.add_bar(bar)

        for i, positions in enumerate(self.generate_positions(max_batch)):
            bar.update(i)

            # for positions in tqdm(self.generate_positions(max_batch), total=num_iter):
            self._scanable.translate.positions = positions
            tensor = self._scanable.get_tensor()

            if potential is not None:
                tensor = tensor.multislice(potential)

            for detector, detections in data.items():
                detections.append(detector.detect(tensor))

        # for detector in data.keys():
        #    data[detector] = tf.reshape(tf.concat(data[detector], axis=0), tf.reshape(self.num_positions, (-1,)))

        if tracker is not None:
            del tracker._output[bar]

        return data

    def generate_scan_positions(self, scan, max_batch=1):
        raise NotImplementedError()


class LineScan(Scan):

    def __init__(self, scanable, detectors, start, end, num_positions=None, sampling=None, endpoint=True):
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

    def show_data(self, detectors=None):
        if detectors is None:
            detectors = self.detectors

        elif not isinstance(detectors, Iterable):
            detectors = (detectors,)

        for detector in detectors:
            data = self.get_data(detector)
            plt.plot(data.numpy(), '.-')

    def get_positions(self):
        return tf.linspace(0., 1, self._num_positions)[:, None] * (self._end - self._start)[None] + self._start[None]


class GridScan(Scan):

    def __init__(self, scanable, detectors, start=None, end=None, num_positions=None, sampling=None, endpoint=False):

        Scan.__init__(self, scanable=scanable, detectors=detectors)

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
            end = scanable.grid.extent

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

    def read_detector(self, detector=None):
        return Image(Scan.read_detector(self, detector)[None], extent=self._end - self._start)

    def get_show_data(self):
        return self.read_detector()

    def get_positions(self):
        x = tf.linspace(self._start[0], self._end[0], self._num_positions[0])
        y = tf.linspace(self._start[1], self._end[1], self._num_positions[1])
        y, x = tf.meshgrid(y, x)
        return tf.stack((tf.reshape(x, (-1,)), tf.reshape(y, (-1,))), axis=1)
