import numpy as np
import tensorflow as tf

from tensorwaves.bases import Grid, Accelerator, TensorFactory
from tensorwaves.utils import complex_exponential, fft_shift, batch_generator, wrapped_slice
from tensorwaves.ctf import CTF, Aperture


def fourier_propagator(k, dz, wavelength):
    """ Fourier space fresnel propagator """
    x = -k * np.pi * wavelength * dz
    return complex_exponential(x)


class TensorWaves(object):

    def __init__(self, tensor, extent=None, sampling=None, energy=None):
        self._tensor = tensor

        gpts = [dim.value for dim in tensor.shape[1:]]

        adjust = {'extent': 'sampling',
                  'gpts': 'sampling',
                  'sampling': 'extent'}

        self.grid = Grid(extent=extent, gpts=gpts, sampling=sampling, adjust=adjust)
        self.accelerator = Accelerator(energy=energy)

    def multislice(self, sliced_potential):
        # if isinstance(potential, SlicedPotential):
        #     sliced_potential = potential
        # elif isinstance(potential, Potential):
        #     sliced_potential = potential.slice(nslices)
        # else:
        #     raise RuntimeError()

        for potential_slice in sliced_potential.slice_generator():
            # potential_slice = sliced_potential[i]
            self.transmit(potential_slice)
            self.propagate(sliced_potential.slice_thickness)

    def fourier_propagator(self, dz):
        kx, ky = self.grid.fftfreq()
        k = ((kx ** 2)[:, None] + (ky ** 2)[None, :])
        return fourier_propagator(k, dz, self.accelerator.wavelength)[None, :, :]

    def transmit(self, potential_slice):
        self._tensor = self._tensor * complex_exponential(self.accelerator.interaction_parameter *
                                                          potential_slice)[None, :, :]

    def propagate(self, dz):
        self._tensor = self._fourier_convolution(self.fourier_propagator(dz))

    def _fourier_convolution(self, propagator):
        return tf.ifft2d(tf.fft2d(self._tensor) * propagator)

    def convolve(self, other, in_place=True):
        if in_place:
            wave = self
        else:
            wave = self.copy()

        other.adopt_grid(wave)
        other.adopt_energy(wave)
        wave._tensor = self._fourier_convolution(other.get_tensor())

        return wave

    def convolve_ctf(self, ctf):
        ctf.adopt_grid(self)
        self._tensor = self._fourier_convolution(ctf._tensor())

    def copy(self):
        tensor_waves = self.__class__(tf.identity(self._tensor))
        tensor_waves.grid = self.grid.copy()
        tensor_waves.accelerator = self.accelerator.copy()
        return tensor_waves


class WaveFactory(TensorFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        TensorFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    def build(self):
        return None


class ProbeWaves(WaveFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, convergence=1, rolloff=0.,
                 parametrization='polar', **kwargs):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)

        self.aperture = Aperture(radius=convergence, rolloff=rolloff)
        self.ctf = CTF(parametrization=parametrization, **kwargs)

        self.aperture.adopt_grid(self)
        self.aperture.adopt_energy(self)
        self.ctf.adopt_grid(self)
        self.ctf.adopt_energy(self)

    def get_tensor(self):
        return fft_shift(tf.fft2d(self.ctf.get_tensor() * tf.cast(self.aperture.get_tensor(), tf.complex64)), (1, 2))

    def build(self):
        return TensorWaves(self.get_tensor(), extent=self.grid.extent, energy=self.accelerator.energy)


class PlaneWaves(WaveFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    def get_tensor(self):
        return tf.ones((1, self.grid.gpts[0], self.grid.gpts[1]), dtype=tf.complex64)

    def build(self):
        return TensorWaves(self.get_tensor(), extent=self.grid.extent, energy=self.accelerator.energy)


class PrismWaves(WaveFactory):

    def __init__(self, cutoff, interpolation, energy=None, gpts=None, extent=None, sampling=None):
        self.cutoff = cutoff
        self.interpolation = interpolation

        WaveFactory.__init__(self, gpts=gpts, extent=extent, sampling=sampling, energy=energy)

    def _coordinates(self):
        n_max = np.ceil(self.cutoff / (self.accelerator.wavelength / self.grid.extent[0] * self.interpolation))
        m_max = np.ceil(self.cutoff / (self.accelerator.wavelength / self.grid.extent[1] * self.interpolation))

        kx = tf.cast(tf.range(-n_max, n_max + 1), tf.float32) / self.grid.extent[0] * self.interpolation
        ky = tf.cast(tf.range(-m_max, m_max + 1), tf.float32) / self.grid.extent[1] * self.interpolation

        mask = tf.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) < (self.cutoff / self.accelerator.wavelength)

        kx, ky = tf.meshgrid(kx, ky)
        kx = tf.boolean_mask(kx, mask)
        ky = tf.boolean_mask(ky, mask)

        x, y = self.grid.linspace()

        return x, y, kx, ky

    def get_tensor(self, x, y, kx, ky):
        return complex_exponential(2 * np.pi * (kx[:, None, None] * x[None, :, None] +
                                                ky[:, None, None] * y[None, None, :]))

    def build(self):
        x, y, kx, ky = self._coordinates()

        return ScatteringMatrix(tensor=self.get_tensor(x, y, kx, ky), kx=kx, ky=ky, interpolation=self.interpolation,
                                energy=self.accelerator.energy, extent=self.grid.extent)

    # def generate(self, max_waves):
    #     x, y, kx, ky = self._coordinates()
    #
    #     for start, size in batch_generator(kx.shape[0].value, max_waves):
    #         end = start + size
    #         yield ScatteringMatrix(tensor=self._tensor(x, y, kx[start:end], kx[start:end]), kx=kx[start:end],
    #                                ky=ky[start:end],
    #                                interpolation_factor=self._interpolation_factor,
    #                                energy=self.energy, extent=self.extent, sampling=self.sampling)
    #
    # def show_probe(self, mode='abs2', space='direct', **kwargs):
    #     self.build().show_probe(mode, space, **kwargs)


class ScatteringMatrix(object):

    def __init__(self, tensor, kx, ky, interpolation, energy=None, extent=None, sampling=None):
        self.interpolation = interpolation
        self.kx = kx
        self.ky = ky

        self.tensor = tensor

        gpts = [dim.value for dim in tensor.shape[1:]]

        adjust = {'extent': 'sampling',
                  'gpts': 'sampling',
                  'sampling': 'extent'}

        self.grid = Grid(extent=extent, gpts=gpts, sampling=sampling, adjust=adjust)
        self.accelerator = Accelerator(energy=energy)

    def probe(self, x, y, parametrization='polar', **kwargs):

        ctf = CTF(energy=self.accelerator.energy)

        alpha_x = self.kx * self.accelerator.wavelength
        alpha_y = self.ky * self.accelerator.wavelength

        alpha2 = alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2
        alpha = tf.sqrt(alpha2)
        alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)

        coefficients = complex_exponential(-2 * np.pi * (self.kx * x + self.ky * y))[:, None, None]

        begin = [0,
                 np.round((x - self.grid.extent[0] / (2 * self.interpolation)) / self.grid.sampling[0]).astype(int),
                 np.round((y - self.grid.extent[1] / (2 * self.interpolation)) / self.grid.sampling[1]).astype(int)]

        size = [self.kx.shape[0].value,
                np.ceil(self.grid.gpts[0] / self.interpolation).astype(int),
                np.ceil(self.grid.gpts[1] / self.interpolation).astype(int)]

        tensor = wrapped_slice(self.tensor, begin, size)

        return tf.reduce_sum(tensor * coefficients, axis=0) / tf.cast(tf.reduce_prod(size[1:]), tf.complex64)

    # def show_probe(self, mode='abs2', space='direct', **kwargs):
    #     tensor = self.probe(*self.extent / 2)
    #     if space == 'fourier':
    #         tensor = tf.fft2d(tensor)
    #     plotutils.show(tensor, self.extent, mode, space, **kwargs)
