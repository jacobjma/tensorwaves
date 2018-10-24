from math import sqrt, pi
import tensorflow as tf

from ase import units

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)
kappa = 4 * pi * eps0 / (2 * pi * units.Bohr * units._e * units.C)

energy2wavelength = lambda x: (
        units._hplanck * units._c / sqrt(x * (2 * units._me * units._c ** 2 / units._e + x)) / units._e * 1e10)

energy2sigma = lambda x: (
        2 * pi * units._me * units.kg * units._e * units.C * energy2wavelength(x) / (
        units._hplanck * units.s * units.J) ** 2)


def complex_exponential(x):
    return tf.complex(tf.cos(x), tf.sin(x))


def fourier_propagator(k, dz, wavelength):
    """ Fourier space fresnel propagator """
    x = -k * pi * wavelength * dz
    return complex_exponential(x)
