import numpy as np
import tensorflow as tf

from tensorwaves.bases import Tensor, Showable, ShowableWithEnergy, HasData, notifying_property
from tensorwaves.utils import freq2angles


class Image(Tensor):

    def __init__(self, tensor, extent=None, sampling=None, grid=None, space='direct'):
        Tensor.__init__(self, tensor, extent=extent, sampling=sampling, grid=grid, space=space)

    # def get_normalization_factor(self):
    #     return tf.reduce_sum(self._tensor) * tf.reduce_prod(self.grid.sampling) * tf.cast(
    #         tf.reduce_prod(self.grid.gpts), tf.float32)
    #
    # def normalize(self, dose):
    #     self._tensor = self._tensor * dose / self.get_normalization_factor()

    def scale_intensity(self, scale):
        self._tensor = self._tensor * scale

    def apply_poisson_noise(self):
        self._tensor = tf.random.poisson(self._tensor, shape=[1])[0]

    def normalize(self):
        mean, stddev = tf.nn.moments(self._tensor, axes=[0, 1, 2])
        self._tensor = (self._tensor - mean) / tf.sqrt(stddev)

    def add_gaussian_noise(self, stddev=1.):
        self._tensor = self._tensor + tf.random.normal(self._tensor.shape, stddev=stddev)

    def clip(self, min=0., max=np.inf):
        self._tensor = tf.clip_by_value(self._tensor, clip_value_min=min, clip_value_max=max)

    def save(self, path):
        np.savez(file=path, tensor=self._tensor.numpy(), extent=self.grid.extent, space=self.space)

    def show(self, i=None, space='direct', scale='linear', **kwargs):
        Showable.show(self, i=i, space=space, mode='real', scale=scale, **kwargs)


class Detector(HasData, ShowableWithEnergy):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, space='direct', save_data=True, grid=None,
                 accelerator=None):
        HasData.__init__(self, save_data=save_data)
        ShowableWithEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid, space=space,
                                    energy=energy, accelerator=accelerator)

        self.grid.register_observer(self)
        self.accelerator.register_observer(self)
        self.register_observer(self)

    # def get_intensity(self, tensor):
    #     return tf.abs(wave.get_tensor()._tensor) ** 2

    def detect(self, wave):
        raise NotImplementedError()

    def get_semiangles(self, return_squared_norm=False, return_azimuth=False):
        return freq2angles(*self._grid.fftfreq(), self._accelerator.wavelength, return_squared_norm, return_azimuth)


class FullFieldDetector(Detector):
    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_data=True, grid=None, accelerator=None):
        Detector.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, space='direct',
                          save_data=save_data, grid=grid, accelerator=accelerator)

    def detect(self, wave):
        intensity = tf.abs(wave.get_tensor().tensorflow()) ** 2

        return Image(intensity, extent=wave.grid.extent)


class WaveDetector(Detector):
    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_data=True, grid=None, accelerator=None):
        Detector.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, space='direct',
                          save_data=save_data, grid=grid, accelerator=accelerator)

    def detect(self, wave):
        return wave


class RingDetector(Detector):

    def __init__(self, inner=None, outer=None, rolloff=0., integrate=True, extent=None, gpts=None, sampling=None,
                 energy=None, save_data=True, grid=None, accelerator=None):
        self._inner = inner
        self._outer = outer
        self._rolloff = rolloff
        self._integrate = integrate

        Detector.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, space='fourier',
                          save_data=save_data, grid=grid, accelerator=accelerator)

    inner = notifying_property('_inner')
    outer = notifying_property('_outer')
    rolloff = notifying_property('_rolloff')

    def _calculate_data(self):
        _, _, alpha2 = self.get_semiangles(return_squared_norm=True)
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
        wave.grid.match(self.grid)
        wave.accelerator.match(self.accelerator)

        intensity = tf.abs(tf.fft2d(wave.get_tensor().tensorflow())) ** 2

        if self._integrate:
            return tf.reduce_sum(intensity * self.get_data(), axis=(1, 2)) / tf.reduce_sum(intensity, axis=(1, 2))
        else:
            return Image(intensity, extent=wave.grid.extent)

    def get_showable_tensor(self, i=None):
        return self.get_data()[None]

    def show(self, i=None, space='fourier', scale='linear', **kwargs):
        Showable.show(self, i=i, space=space, mode='real', scale=scale, **kwargs)

    def _create_tensor(self):
        pass
