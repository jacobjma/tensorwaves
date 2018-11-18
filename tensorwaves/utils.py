import tensorflow as tf
import numpy as np

from ase import units
from IPython.display import clear_output

EPS = 1e-12

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

energy2wavelength = lambda x: (
        units._hplanck * units._c / np.sqrt(x * (2 * units._me * units._c ** 2 / units._e + x)) / units._e * 1e10)

energy2sigma = lambda x: (
        2 * np.pi * units._me * units.kg * units._e * units.C * energy2wavelength(x) / (
        units._hplanck * units.s * units.J) ** 2)


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


def linspace_no_endpoint(n, l):
    return tf.lin_space(0., l - l / tf.cast(n, tf.float32), n)


def fftfreq(n, h):
    N = (n - 1) // 2 + 1
    p1 = tf.lin_space(0., N - 1, N)
    p2 = tf.lin_space(-float(n // 2), -1, n // 2)
    return tf.concat((p1, p2), axis=0) / (n * h)


def fft_shift(tensor, axes):
    shift = [tensor.shape[axis].value // 2 for axis in axes]
    return tf.manip.roll(tensor, shift, axes)


def wrapped_slice(tensor, begin, size):
    shift = [-x for x in begin]
    tensor = tf.manip.roll(tensor, shift, list(range(len(begin))))
    return tf.slice(tensor, [0] * len(begin), size)





class ProgressBar(object):

    def __init__(self, num_iter, units='', description='', update_every=5, disable=False):

        self._num_iter = num_iter
        self._units = units
        self._description = description
        self._update_every = update_every
        self._disable = disable

        self._intervals = 100 // update_every
        self._last_update = None

    def print(self, i):

        if not self._disable:
            self._print(i)

    def _print(self, i):

        p = int((i + 1) / self._num_iter * self._intervals) * self._update_every

        if p != self._last_update:
            self._last_update = p
            progress_bar = ('|' * (p // self._update_every)).ljust(self._intervals)
            print('{} [{}] {}/{} {}'.format(self._description, progress_bar, i + 1, self._num_iter, self._units))
            clear_output(wait=True)


def bar(itrble, num_iter=None, **kwargs):
    '''Simple progress bar. '''

    if num_iter is None:
        num_iter = len(itrble)

    progress_bar = ProgressBar(num_iter, **kwargs)

    for i, j in enumerate(itrble):
        yield j

        progress_bar.print(i)
