from __future__ import print_function
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from ase import units

EPS = 1e-12










def freq2angles(kx, ky, wavelength, return_squared_norm=False, return_azimuth=False):
    alpha_x = kx * wavelength
    alpha_y = ky * wavelength

    if return_squared_norm & return_azimuth:
        return (alpha_x, alpha_y, alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2,
                tf.atan2(alpha_x[:, None], alpha_y[None, :]))

    elif return_squared_norm:
        return alpha_x, alpha_y, alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2

    elif return_azimuth:

        return alpha_x, alpha_y, tf.atan2(alpha_x[:, None], alpha_y[None, :])
    else:

        return alpha_x, alpha_y


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
    dt = tf.log(stop / start) / (n - 1)
    return start * tf.exp(dt * tf.lin_space(0., n - 1, n))


def complex_exponential(x):
    return tf.complex(tf.cos(x), tf.sin(x))


def fourier_propagator(k, dz, wavelength):
    """ Fourier space fresnel propagator """
    x = -k * np.pi * wavelength * dz
    return complex_exponential(x)


def cell_is_rectangular(cell, tol=1e-12):
    return np.all(np.abs(cell[~np.eye(cell.shape[0], dtype=bool)]) < tol)





def fftfreq2d(gpts, sampling):
    value = []
    for n, h in zip(gpts, sampling):
        value += [fftfreq(n, h)]
    return value


def fft_shift(tensor, axes):
    shift = [tensor.shape[axis].value // 2 for axis in axes]
    return tf.manip.roll(tensor, shift, axes)


def wrapped_slice(tensor, begin, size):
    shift = [-x for x in begin]
    tensor = tf.manip.roll(tensor, shift, list(range(len(begin))))
    return tf.slice(tensor, [0] * len(begin), size)


class ProgressTracker(object):

    def __init__(self, bars=None):
        if bars is None:
            bars = []

        self._output = OrderedDict(zip(bars, [''] * len(bars)))

    def add_bar(self, bar):
        bar._tracker = self
        self._output[bar] = ''

    def notify(self, bar):
        percentage = bar.get_percentage()
        updated = False
        if percentage != bar._last_update:
            bar._last_update = percentage
            self._output[bar] = bar.get_output(percentage)
            updated = True

        if updated:
            for bar, bar_out in self._output.items():
                print(bar_out)
            clear_output(wait=True)


class ProgressBar(object):

    def __init__(self, num_iter, units='', description='', update_every=2, disable=False, tracker=None):

        self._num_iter = num_iter
        self._units = units
        self._description = description
        self._update_every = update_every
        self._disable = disable

        self._intervals = 100 // update_every
        self._last_update = None

        self._i = 0
        self._tracker = tracker

    def get_percentage(self):
        percentage = int((self._i + 1) / float(self._num_iter) * self._intervals) * self._update_every
        return percentage

    def get_output(self, percentage):
        progress_bar = ('|' * (percentage // self._update_every)).ljust(self._intervals)
        output = '{} [{}] {}/{} {}'.format(self._description, progress_bar, self._i + 1, self._num_iter, self._units)
        return output

    def update(self, i):

        self._i = i

        if self._tracker:
            self._tracker.notify(self)
        else:
            if not self._disable:
                self._print()

    def _print(self):
        percentage = self.get_percentage()
        if percentage != self._last_update:
            self._last_update = percentage
            output = self.get_output(percentage)
            print(output)
            clear_output(wait=True)


def bar(itrble, num_iter=None, **kwargs):
    '''Simple progress bar. '''

    if num_iter is None:
        num_iter = len(itrble)

    progress_bar = ProgressBar(num_iter, **kwargs)

    for i, j in enumerate(itrble):
        yield j
        progress_bar.update(i)
