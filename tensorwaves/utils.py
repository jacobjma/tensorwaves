import tensorflow as tf
import numpy as np

from ase import units

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)

energy2wavelength = lambda x: (
        units._hplanck * units._c / np.sqrt(x * (2 * units._me * units._c ** 2 / units._e + x)) / units._e * 1e10)

energy2sigma = lambda x: (
        2 * np.pi * units._me * units.kg * units._e * units.C * energy2wavelength(x) / (
        units._hplanck * units.s * units.J) ** 2)


def log_grid(start, stop, n):
    dt = tf.log(stop / start) / (n - 1)
    return start * tf.exp(dt * tf.lin_space(0., n - 1, n))


def complex_exponential(x):
    return tf.complex(tf.cos(x), tf.sin(x))


def fourier_propagator(k, dz, wavelength):
    """ Fourier space fresnel propagator """
    x = -k * np.pi * wavelength * dz
    return complex_exponential(x)


def cell_is_rectangular(cell, tol=1e-14):
    return np.all(np.abs(cell[~np.eye(cell.shape[0], dtype=bool)]) < tol)


def linspace_no_endpoint(n, l):
    return tf.lin_space(0., l - l / tf.cast(n, tf.float32), n)


def fftfreq(n, h):
    N = (n - 1) // 2 + 1
    p1 = tf.lin_space(0., N - 1, N)
    p2 = tf.lin_space(-float(n // 2), -1, n // 2)
    return tf.concat((p1, p2), axis=0) / (n * h)
