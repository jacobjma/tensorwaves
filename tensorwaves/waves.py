import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorwaves import plotutils
from tensorwaves.bases import TensorBase, FactoryBase
from tensorwaves.ctf import CTF
from tensorwaves import utils


class TensorWaves(TensorBase):

    def __init__(self, tensor, energy=None, extent=None, sampling=None):
        super().__init__(tensor=tensor, energy=energy, extent=extent, sampling=sampling)

    @property
    def n_waves(self):
        return self.shape[0]

    def __len__(self):
        return self.shape[0]

    def fourier_propagator(self, dz):
        kx, ky = self.fftfreq()
        k = ((kx ** 2)[:, None] + (ky ** 2)[None, :])
        return utils.fourier_propagator(k, dz, self.wavelength)[None, :, :]

    def propagate(self, dz):
        self._tensor = self._fourier_convolution(self.fourier_propagator(dz))

    def _fourier_convolution(self, propagator):
        return tf.ifft2d(tf.fft2d(self._tensor) * propagator)

    def apply_ctf(self, ctf=None, **kwargs):
        if ctf is not None:
            self.is_compatible(ctf)
        else:
            ctf = CTF(**kwargs)
        ctf.adapt(self)
        self._tensor = self._fourier_convolution(ctf._tensor()[None, :, :])

    def show_image(self):
        ax, mapable = plotutils.show_image(np.abs(self._tensor[0].numpy()) ** 2)
        plt.colorbar(mapable)

    def __repr__(self):
        if self.n_waves > 1:
            plural = 's'
        else:
            plural = ''

        return '{} tensor wave{}\n'.format(self.n_waves, plural) + super().__repr__()


class WaveFactory(FactoryBase):
    def __init__(self, energy=None, n_waves=None, gpts=None, extent=None, sampling=None):
        super().__init__(gpts=gpts, extent=extent, sampling=sampling, energy=energy)

        self.n_waves = n_waves

    def __len__(self):
        return self.n_waves

    def check_buildable(self):
        if self.gpts is None:
            raise RuntimeError('the grid is not defined')
        if self.n_waves is None:
            raise RuntimeError('the number of waves is not defined')


class PlaneWaves(WaveFactory):

    def __init__(self, energy=None, n_waves=1, gpts=None, extent=None, sampling=None):
        super().__init__(energy=energy, n_waves=n_waves, gpts=gpts, extent=extent, sampling=sampling)

    def _tensor(self):
        return tf.ones((self.n_waves,) + self.gpts, dtype=tf.complex64)

    def build(self):
        self.check_buildable()
        return TensorWaves(tensor=self._tensor(), energy=self.energy, extent=self.extent, sampling=self.sampling)

    def __repr__(self):
        if self.n_waves > 1:
            plural = 's'
        else:
            plural = ''

        return '{} plane wave{}\n'.format(self.n_waves, plural) + super().__repr__()


class PrismWaves(WaveFactory):
    pass


class ProbeWaves(WaveFactory):
    pass
