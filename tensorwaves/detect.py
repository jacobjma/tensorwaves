import numpy as np
import tensorflow as tf

from tensorwaves.bases import TensorFactory, HasGrid, HasEnergy, notifying_property
from tensorwaves.transfer import squared_norm
from tensorwaves.image import Image


class Detector(HasGrid):

    def __init__(self, extent=None, gpts=None, sampling=None):
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)

    def detect(self, wave):
        raise NotImplementedError()


class DetectorWithEnergy(Detector, HasEnergy):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        Detector.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        HasEnergy.__init__(self, energy=energy)

    def semiangles(self):
        kx, ky = self._grid.fftfreq()
        wavelength = self.wavelength
        return kx * wavelength, ky * wavelength

    def match(self, other):
        try:
            self._grid.match(other._grid)
        except AttributeError:
            pass

        try:
            self._energy.match(other._energy)
        except AttributeError:
            pass


class RingDetector(DetectorWithEnergy, TensorFactory):

    def __init__(self, inner, outer, rolloff=0., integrate=True, extent=None, gpts=None, sampling=None,
                 energy=None, save_tensor=True):
        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff
        self._integrate = integrate

        DetectorWithEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)
        TensorFactory.__init__(self, save_tensor=save_tensor)

    inner = notifying_property('_inner')
    outer = notifying_property('_outer')
    rolloff = notifying_property('_rolloff')

    def _calculate_tensor(self):

        alpha2 = squared_norm(*self.semiangles())
        alpha = tf.sqrt(alpha2)

        if self.rolloff > 0.:
            tensor_outer = .5 * (1 + tf.cos(np.pi * (alpha - self.outer) / self.rolloff))
            tensor_outer *= tf.cast(alpha < self.outer + self.rolloff, tf.float32)
            tensor_outer = tf.where(alpha > self.outer, tensor_outer, tf.ones(alpha.shape))

            tensor_inner = .5 * (1 + tf.cos(np.pi * (self.inner - alpha) / self.rolloff))
            tensor_inner *= tf.cast(alpha > self.inner - self.rolloff, tf.float32)
            tensor_inner = tf.where(alpha < self.inner, tensor_inner, tf.ones(alpha.shape))
            tensor = tensor_inner * tensor_outer

        else:
            tensor = tf.cast((alpha > self._inner) & (alpha < self._outer), tf.float32)

        return tensor

    def detect(self, wave):
        self.match(wave)

        intensity = tf.abs(tf.fft2d(wave.get_tensor().tensorflow())) ** 2

        if self._integrate:
            return tf.reduce_sum(intensity * self.get_tensor(), axis=(1, 2)) / tf.reduce_sum(intensity, axis=(1, 2))
        else:
            return Image(intensity, extent=wave.grid.extent)
