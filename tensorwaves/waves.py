import numpy as np
import tensorflow as tf

#from tensorwaves.bases_old import TensorBase, FactoryBase
from tensorwaves.ctf import CTF, Aperture
from tensorwaves import utils, plotutils
from tensorwaves.potentials import Potential, SlicedPotential


class Simulateable(TensorBase):

    def __init__(self, tensor, energy=None, extent=None, sampling=None):
        super().__init__(tensor=tensor, energy=energy, extent=extent, sampling=sampling)

    def __len__(self):
        return self.shape[0]

    def multislice(self, potential, nslices=None):

        if isinstance(potential, SlicedPotential):
            sliced_potential = potential
        elif isinstance(potential, Potential):
            sliced_potential = potential.slice(nslices)
        else:
            raise RuntimeError()

        for i in utils.bar(range(potential.nslices)):
            potential_slice = sliced_potential[i]
            self.transmit(potential_slice.tensor())
            self.propagate(potential_slice.thickness)

    def fourier_propagator(self, dz):
        kx, ky = self.fftfreq()
        k = ((kx ** 2)[:, None] + (ky ** 2)[None, :])
        return utils.fourier_propagator(k, dz, self.wavelength)[None, :, :]

    def transmit(self, potential_slice):
        self._tensor = self._tensor * utils.complex_exponential(self.interaction_parameter *
                                                                potential_slice)[None, :, :]

    def propagate(self, dz):
        self._tensor = self._fourier_convolution(self.fourier_propagator(dz))

    def _fourier_convolution(self, propagator):
        return tf.ifft2d(tf.fft2d(self._tensor) * propagator)


class TensorWaves(Simulateable):

    def __init__(self, tensor, energy=None, extent=None, sampling=None):
        super().__init__(tensor=tensor, energy=energy, extent=extent, sampling=sampling)

    @property
    def n_waves(self):
        return self.shape[0]

    def apply_ctf(self, ctf=None, **kwargs):
        if ctf is not None:
            self.is_compatible(ctf)
        else:
            ctf = CTF(**kwargs)
        ctf.adapt(self)
        self._tensor = self._fourier_convolution(ctf._tensor()[None, :, :])

    def __repr__(self):
        if self.n_waves > 1:
            plural = 's'
        else:
            plural = ''

        return '{} tensor wave{}\n'.format(self.n_waves, plural) + super().__repr__()


class ScatteringMatrix(Simulateable):

    def __init__(self, tensor, kx, ky, interpolation_factor, energy=None, extent=None, sampling=None):
        super().__init__(tensor=tensor, energy=energy, extent=extent, sampling=sampling)

        self._interpolation_factor = interpolation_factor
        self._kx = kx
        self._ky = ky

    def probe(self, x, y):
        coefficients = utils.complex_exponential(-2 * np.pi * (self._kx * x + self._ky * y))[:, None, None]

        begin = [0,
                 np.round((x - self.extent[0] / (2 * self._interpolation_factor)) / self.sampling[0]).astype(int),
                 np.round((y - self.extent[1] / (2 * self._interpolation_factor)) / self.sampling[1]).astype(int)]

        size = [self._kx.shape[0].value,
                np.ceil(self.gpts[0] / self._interpolation_factor).astype(int),
                np.ceil(self.gpts[1] / self._interpolation_factor).astype(int)]

        tensor = utils.wrapped_slice(self._tensor, begin, size)
        return tf.reduce_sum(tensor * coefficients, axis=0) / tf.cast(tf.reduce_prod(size[1:]), tf.complex64)

    def show_probe(self, mode='abs2', space='direct', **kwargs):
        tensor = self.probe(*self.extent / 2)
        if space == 'fourier':
            tensor = tf.fft2d(tensor)
        plotutils.show(tensor, self.extent, mode, space, **kwargs)


class WaveFactory(FactoryBase):
    def __init__(self, energy=None, n_waves=None, gpts=None, extent=None, sampling=None):
        super().__init__(gpts=gpts, extent=extent, sampling=sampling, energy=energy)

        self._n_waves = n_waves

    def __len__(self):
        return self._n_waves

    def build(self):
        raise RuntimeError()

    @property
    def n_waves(self):
        return self._n_waves

    def check_buildable(self):
        if self.gpts is None:
            raise RuntimeError('the grid is not defined')
        if self._n_waves is None:
            raise RuntimeError('the number of waves is not defined')


class PlaneWaves(WaveFactory):

    def __init__(self, n_waves=1, energy=None, gpts=None, extent=None, sampling=None):
        super().__init__(energy=energy, n_waves=n_waves, gpts=gpts, extent=extent, sampling=sampling)

    def _tensor(self):
        return tf.ones((self.n_waves,) + tuple(self.gpts), dtype=tf.complex64)

    def build(self):
        self.check_buildable()
        return TensorWaves(tensor=self._tensor(), energy=self.energy, extent=self.extent, sampling=self.sampling)

    def __repr__(self):
        if self.n_waves > 1:
            plural = 's'
        else:
            plural = ''

        return '{} plane wave{}\n'.format(self.n_waves, plural) + super().__repr__()


class ProbeWaves(WaveFactory):

    def __init__(self, convergenge_angle, rolloff=None, positions=None, parametrization='polar', parameters=None,
                 energy=None, gpts=None, extent=None, sampling=None, **kwargs):

        if positions is not None:
            n_waves = len(positions)
        else:
            n_waves = 1

        super().__init__(energy=energy, n_waves=n_waves, gpts=gpts, extent=extent, sampling=sampling)

        if rolloff is None:
            rolloff = 1e-2 * convergenge_angle  # small rolloff to improve numerical stability

        self._aperture = Aperture(convergenge_angle, rolloff)

        self._ctf = CTF(parametrization=parametrization, parameters=parameters, **kwargs)

        self._aperture.adapt(self)
        self._ctf.adapt(self)

        self._positions = positions

    def _tensor(self):
        tensor = utils.fft_shift(tf.ifft2d(tf.cast(self._aperture._tensor(), tf.complex64) * self._ctf._tensor()),
                                 (1, 2))
        return tensor / tf.cast(tf.reduce_sum(tf.abs(tensor) ** 2), tf.complex64)

    def build(self):
        self.check_buildable()
        return TensorWaves(tensor=self._tensor(), energy=self.energy, extent=self.extent, sampling=self.sampling)

    def __repr__(self):
        if self.n_waves is None:
            return 'probe template\n' + super().__repr__()
        else:
            if self.n_waves > 1:
                plural = 's'
            else:
                plural = ''
            return '{} probe{}\n'.format(self.n_waves, plural) + super().__repr__()


class PrismWaves(WaveFactory):

    def __init__(self, cutoff_angle, interpolation_factor, energy=None, gpts=None, extent=None, sampling=None):
        self._cutoff_angle = cutoff_angle
        self._interpolation_factor = interpolation_factor

        super().__init__(energy=energy, gpts=gpts, extent=extent, sampling=sampling)

    def _coordinates(self):
        n_max = np.ceil(self._cutoff_angle / (self.wavelength / self.extent[0] * self._interpolation_factor))
        m_max = np.ceil(self._cutoff_angle / (self.wavelength / self.extent[1] * self._interpolation_factor))

        kx = tf.cast(tf.range(-n_max, n_max + 1), tf.float32) / self.extent[0] * self._interpolation_factor
        ky = tf.cast(tf.range(-m_max, m_max + 1), tf.float32) / self.extent[1] * self._interpolation_factor

        mask = tf.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) < (self._cutoff_angle / self.wavelength)

        kx, ky = tf.meshgrid(kx, ky)
        kx = tf.boolean_mask(kx, mask)
        ky = tf.boolean_mask(ky, mask)

        x, y = self.linspace()

        return x, y, kx, ky

    def _tensor(self, x, y, kx, ky):
        return utils.complex_exponential(2 * np.pi * (kx[:, None, None] * x[None, :, None] +
                                                      ky[:, None, None] * y[None, None, :]))

    def build(self):
        x, y, kx, ky = self._coordinates()

        return ScatteringMatrix(tensor=self._tensor(x, y, kx, ky), kx=kx, ky=ky,
                                interpolation_factor=self._interpolation_factor, energy=self.energy,
                                extent=self.extent, sampling=self.sampling)

    def generate(self, max_waves):
        x, y, kx, ky = self._coordinates()

        for start, size in utils.batch_generator(kx.shape[0].value, max_waves):
            end = start + size
            yield ScatteringMatrix(tensor=self._tensor(x, y, kx[start:end], kx[start:end]), kx=kx[start:end],
                                   ky=ky[start:end],
                                   interpolation_factor=self._interpolation_factor,
                                   energy=self.energy, extent=self.extent, sampling=self.sampling)

    def show_probe(self, mode='abs2', space='direct', **kwargs):
        self.build().show_probe(mode, space, **kwargs)
