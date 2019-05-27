from collections import OrderedDict

import numpy as np
import tensorflow as tf
from IPython.display import clear_output


def batch_generator(n_items, max_batch_size):
    n_batches = (n_items + (-n_items % max_batch_size)) // max_batch_size
    batch_size = (n_items + (-n_items % n_batches)) // n_batches

    batch_start = 0
    while 1:
        batch_end = batch_start + batch_size
        if batch_end >= n_items:
            yield batch_start, n_items - batch_end + batch_size
            break
        else:
            yield batch_start, batch_size

        batch_start = batch_end


def log_grid(start, stop, n):
    dt = tf.math.log(stop / start) / (n - 1)
    return start * tf.exp(dt * tf.linspace(0., n - 1, n))


def complex_exponential(x):
    return tf.complex(tf.cos(x), tf.sin(x))


def squared_norm(x, y):
    return x[:, None] ** 2 + y[None, :] ** 2


def angle(x, y):
    return tf.atan2(x[:, None], y[None, :])


def cell_is_rectangular(cell, tol=1e-12):
    return np.all(np.abs(cell[~np.eye(cell.shape[0], dtype=bool)]) < tol)


def fourier_propagator(k, dz, wavelength):
    x = -k * np.pi * wavelength * dz
    return complex_exponential(x)


def fft_shift(tensor, axes):
    shift = [tensor.shape[axis] // 2 for axis in axes]
    return tf.roll(tensor, shift, axes)


class ProgressBar(object):

    def __init__(self, num_iter, units='', description='', update_every=2, disable=False):

        self._num_iter = num_iter
        self._units = units
        self._description = description
        self._update_every = update_every
        self._disable = disable

        self._intervals = 100 // update_every
        self._last_update = None

        self._i = 0

    def get_percentage(self):
        percentage = int((self._i + 1) / float(self._num_iter) * self._intervals) * self._update_every
        return percentage

    def get_output(self, percentage):
        progress_bar = ('|' * (percentage // self._update_every)).ljust(self._intervals)
        output = '{} [{}] {}/{} {}'.format(self._description, progress_bar, self._i + 1, self._num_iter, self._units)
        return output

    def update(self, i):
        self._i = i
        if not self._disable:
            self.print()

    def print(self):
        percentage = self.get_percentage()
        if percentage != self._last_update:
            self._last_update = percentage
            output = self.get_output(percentage)
            print(output)
            clear_output(wait=True)


def create_progress_bar(itrble, num_iter=None, **kwargs):
    '''Simple progress bar. '''

    if num_iter is None:
        num_iter = len(itrble)

    progress_bar = ProgressBar(num_iter, **kwargs)

    progress_bar.update(-1)
    for i, j in enumerate(itrble):
        progress_bar.update(i)
        yield j
