import numpy as np
import tensorflow as tf

from tensorwaves.bases import TensorFactory, HasEnergy, notifying_property, complex_exponential, Observable
from tensorwaves.transfer import PolarAberrations


class PrismCoefficients(TensorFactory, HasEnergy, Observable):

    def __init__(self, kx, ky, energy=None, save_tensor=True):
        self._kx = kx
        self._ky = ky

        HasEnergy.__init__(self, energy=energy)
        TensorFactory.__init__(self, save_tensor=save_tensor)
        Observable.__init__(self)

        self.register_observer(self)

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky


class PrismAperture(PrismCoefficients):

    def __init__(self, kx, ky, radius=np.inf, rolloff=0., energy=None, save_tensor=True):

        self._radius = radius
        self._rolloff = rolloff

        PrismCoefficients.__init__(self, kx=kx, ky=ky, energy=energy, save_tensor=save_tensor)

    radius = notifying_property('_radius')
    rolloff = notifying_property('_rolloff')

    def _calculate_tensor(self):
        alpha_x = self.kx * self.wavelength
        alpha_y = self.ky * self.wavelength

        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)

        if self.rolloff > 0.:
            tensor = .5 * (1 + tf.cos(np.pi * (alpha - self.radius) / self.rolloff))
            tensor *= tf.cast(alpha < self.radius + self.rolloff, tf.float32)
            tensor = tf.where(alpha > self.radius, tensor, tf.ones(alpha.shape))
        else:
            tensor = tf.cast(alpha < self.radius, tf.float32)

        return tensor


class PrismAberration(PrismCoefficients):

    def __init__(self, kx, ky, energy=None, save_tensor=True, parametrization='polar', **kwargs):

        if parametrization.lower() == 'polar':
            self._parameters = PolarAberrations(**kwargs)

        else:
            raise RuntimeError()

        PrismCoefficients.__init__(self, kx=kx, ky=ky, energy=energy, save_tensor=save_tensor)

        self._parameters.register_observer(self)

    @property
    def parameters(self):
        return self._parameters

    def _calculate_tensor(self):
        alpha_x = self.kx * self.wavelength
        alpha_y = self.ky * self.wavelength
        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)
        phi = tf.atan2(alpha_x, alpha_y)
        #ssss
        tensor = self.parameters(alpha=alpha, alpha2=alpha2, phi=phi)

        return complex_exponential(- 2 * np.pi / self.wavelength * tensor)


class PrismTranslate(PrismCoefficients):

    def __init__(self, kx, ky, positions=None, energy=None, save_tensor=True):

        PrismCoefficients.__init__(self, kx=kx, ky=ky, energy=energy, save_tensor=save_tensor)

        if positions is None:
            positions = (0., 0.)

        self._positions = self._validate_positions(positions)

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        old = self.positions
        self._positions = self._validate_positions(positions)
        change = np.any(self._positions != old)
        self.notify_observers({'name': '_positions', 'old': old, 'new': positions, 'change': change})

    def _validate_positions(self, positions):
        if isinstance(positions, (np.ndarray, list, tuple)):
            positions = np.array(positions, dtype=np.float32)
            if positions.shape == (2,):
                positions = positions[None, :]

        else:
            raise RuntimeError('')

        return positions

    def _calculate_tensor(self):

        tensor = complex_exponential(-2 * np.pi * (self.kx * self.positions[:, 0] + self.ky * self.positions[:, 1]))

        return tf.cast(tensor, tf.complex64)
