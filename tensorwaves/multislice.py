import tensorflow as tf
from math import pi
import utils

from waves import Waves


def realspace_progagator(grid, wavelength):
    """ Real space fresnel propagator """
    x, y = grid.x_axis.grid(), grid.y_axis.grid()
    r = pi / (wavelength * grid.hz) * ((x ** 2)[:, None] + (y ** 2)[None, :])
    return tf.complex(tf.sin(r), - tf.cos(r)) / (wavelength * grid.hz)


def fourier_propagator(axes, wavelength):
    """ Fourier space fresnel propagator """
    kx, ky = axes.x_axis.frequencies(), axes.y_axis.frequencies()
    x = -((kx ** 2)[:, None] + (ky ** 2)[None, :]) * pi * wavelength * axes.hz
    return utils.complex_exponential(x)


def propagate(wave, p):
    """ Convolution with Fresnel propagator using the Fourier convolution theorem"""
    return tf.ifft2d(tf.fft2d(wave) * p[None, ...])


def multislice(potential, waves, h_slice=None, n_slice=None, device=None):
    if waves.axes.lst != potential.axes.lst[:2]:
        raise RuntimeError()

    if device is None:
        device = utils.get_device()

    with tf.device(device):
        chunk = potential.project_potential_chunk(h_slice=h_slice, n_slice=n_slice)

        p = fourier_propagator(chunk.axes, waves.wavelengths)
        new_tensor = waves.tensor

        for i in range(chunk.axes.nz):
            x = waves.interaction_parameters * chunk.array[..., i]
            new_tensor = propagate(new_tensor, p) * tf.complex(tf.cos(x), tf.sin(x))[None, ...]

    return Waves(waves.axes, energies=waves.energies, tensor=new_tensor)
