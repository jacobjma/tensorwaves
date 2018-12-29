import tensorflow as tf

from tensorwaves.bases import Tensor, HasGrid, HasAccelerator, HasData
from tensorwaves.plotutils import show_array
from tensorwaves.utils import freq2angles


class Image(Tensor):

    def __init__(self, tensor, extent=None, sampling=None, space='direct'):
        Tensor.__init__(self, tensor, extent=extent, sampling=sampling, space=space)

    def get_normalization_factor(self):
        return tf.reduce_sum(self._tensor) * tf.reduce_prod(self.grid.sampling) * tf.reduce_prod(self.grid.gpts)

    def normalize(self, dose):
        self._tenor = self._tensor * dose / self.get_normalization_factor()

    def add_poisson_noise(self, in_place=True):
        self._tensor = tf.random.poisson(self._tensor)

    def show(self, **kwargs):
        show_array(self.numpy(), extent=self.grid.extent, space=self.grid.space, display_space=self.grid.space,
                   mode='real',
                   **kwargs)


class Detector(HasData, HasGrid, HasAccelerator):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, space='direct', save_data=True, grid=None,
                 accelerator=None):
        HasData.__init__(self, save_data=save_data)
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, space=space, grid=grid)
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        self.grid.register_observer(self)
        self.accelerator.register_observer(self)

    # def get_intensity(self, tensor):
    #     return tf.abs(wave.get_tensor()._tensor) ** 2

    def detect(self, wave):
        raise NotImplementedError()

    def get_semiangles(self, return_squared_norm=False, return_azimuth=False):
        return freq2angles(*self._grid.fftfreq(), self._accelerator.wavelength, return_squared_norm, return_azimuth)


class RingDetector(Detector):

    def __init__(self, inner=None, outer=None, integrate=True, extent=None, gpts=None, sampling=None, energy=None,
                 save_data=True, grid=None, accelerator=None):
        self._inner = inner
        self._outer = outer
        self._integrate = integrate

        Detector.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, space='fourier',
                          save_data=save_data, grid=grid, accelerator=accelerator)

    def _calculate_data(self):
        _, _, alpha2 = self.get_semiangles(return_squared_norm=True)
        alpha = tf.sqrt(alpha2)
        return (alpha > self._inner) & (alpha < self._outer)

    def detect(self, wave):
        wave.grid.match(self.grid)
        wave.accelerator.match(self.accelerator)

        intensity = tf.abs(tf.fft2d(wave.get_tensor().tensorflow())) ** 2 * tf.cast(self.get_data(), tf.float32)

        if self._integrate:
            return tf.reduce_sum(intensity, axis=(1, 2))
        else:
            return Image(intensity, extent=wave.grid.extent)

    def get_tensor(self):
        pass

    def show(self):
        pass

    def _create_tensor(self):
        pass
