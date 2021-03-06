from numbers import Number

import numpy as np
import tensorflow as tf

from tensorwaves.bases import TensorWithGrid
from tensorwaves.utils import BatchGenerator, create_progress_bar


class Scan(object):

    def __init__(self):
        self._data = None

    def get_positions(self):
        raise NotImplementedError('')

    @property
    def num_positions(self):
        raise NotImplementedError('')

    def generate_positions(self, max_batch, progress_bar=True):
        num_iter = (np.prod(self.num_positions) + max_batch - 1) // max_batch

        positions = self.get_positions()

        generator = BatchGenerator(positions.shape[0], max_batch)

        for start, stop in create_progress_bar(generator.generate(),
                                               num_iter=generator.n_batches,
                                               description='Scanning',
                                               disable=not progress_bar):
            yield positions[start:start + stop]


class CustomScan(Scan):
    def __init__(self, positions):
        Scan.__init__(self)
        self._positions = positions

    @property
    def num_positions(self):
        return len(self._positions)

    def get_positions(self):
        return self._positions

    def numpy(self):
        detector = next(iter(self._data.keys()))
        return np.hstack([value.numpy() for value in self._data[detector]])


class LineScan(Scan):

    def __init__(self, start, end, num_positions=None, sampling=None, endpoint=True):
        Scan.__init__(self)

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

    @property
    def length(self):
        return np.linalg.norm((self._start - self._end), axis=0)

    @property
    def sampling(self):
        return self.length / self.num_positions

    def get_positions(self):
        return np.linspace(0., 1, self.num_positions)[:, None] * (self.end - self.start)[None] + self.start[None]

    def numpy(self, detector=None):
        if detector is None:
            detector = next(iter(self._data.keys()))

        # print(self._data[detector])

        return np.hstack(tf.convert_to_tensor(tf.stack(self._data[detector])).numpy())


class GridScan(Scan):

    def __init__(self, start, end, num_positions=None, sampling=None,
                 endpoint=False):

        Scan.__init__(self)

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

        self._start = np.array(start, dtype=np.float32)
        self._end = np.array(end, dtype=np.float32)

        if (num_positions is None) & (sampling is not None):
            self._num_positions = np.ceil((np.abs(self._start - self._end)) / sampling).astype(np.int)
            # if not endpoint:
            #    self._num_positions -= 1

        elif num_positions is not None:
            self._num_positions = validate(num_positions, dtype=np.int32)

        else:
            raise TypeError('pass either argument: \'num_positions\' or \'sampling\'')

        if not endpoint:
            self._end = self._end - np.abs(self._start - self._end) / np.array(self._num_positions)

    @property
    def num_positions(self):
        return self._num_positions

    @property
    def sampling(self):
        return np.array([(self._end[0] - self._start[0]) / (self._num_positions[0] - 1),
                         (self._end[1] - self._start[1]) / (self._num_positions[1] - 1)])

    def get_x_positions(self):
        return np.linspace(self._start[0], self._end[0], self._num_positions[0])

    def get_y_positions(self):
        return np.linspace(self._start[1], self._end[1], self._num_positions[1])

    def get_positions(self):
        x = np.linspace(self._start[0], self._end[0], self._num_positions[0])
        y = np.linspace(self._start[1], self._end[1], self._num_positions[1])
        y, x = np.meshgrid(y, x)

        return np.stack((np.reshape(x, (-1,)), np.reshape(y, (-1,))), axis=1)

    def image(self, detector=None):
        if detector is None:
            detector = next(iter(self._data.keys()))

        tensor = tf.reshape(tf.concat(self._data[detector], 0), self._num_positions)

        return TensorWithGrid(tensor[None], extent=self._end - self._start)

    def numpy(self):
        return self.image().numpy()
        # np.hstack(tf.convert_to_tensor(self._data[detector]).numpy()).reshape(self._num_positions)

    def split(self, n):
        positions = self.get_positions()

        N = positions.shape[0]
        m = N // n
        scanners = []

        i = 0
        for i in range(n - 1):
            scanners.append(CustomScan(positions=positions[i * m:(i + 1) * m]))

        scanners.append(CustomScan(positions=positions[(i + 1) * m:]))

        return scanners
