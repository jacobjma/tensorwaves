import numpy as np
import tensorflow as tf

from tensorwaves.bases import TensorFactory, HasEnergy, notifying_property, Observable
from tensorwaves.transfer import PolarAberrations
from tensorwaves.utils import complex_exponential


class PrismCoefficients(TensorFactory, Observable):

    def __init__(self, kx, ky, save_tensor=True):
        self._kx = kx
        self._ky = ky

        TensorFactory.__init__(self, save_tensor=save_tensor)
        Observable.__init__(self)

        self.register_observer(self)

    def check_is_defined(self):
        pass

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky


class PrismCustomCoefficient(PrismCoefficients):

    def __init__(self, kx, ky, coeffiecients, save_tensor=True):
        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)
        self._coefficients = coeffiecients

    coefficients = notifying_property('_coeffiecients')


def gaussian(r2, a, b):
    return a * tf.exp(-r2 / (b ** 2))


def gaussian_derivative(r, r2, a, b):
    return - 2 * a * 1 / b ** 2 * r * tf.exp(-r2 / b ** 2)


def soft_gaussian(r, r2, a, b, r_cut):
    return gaussian(r2, a, b) - gaussian(r_cut ** 2, a, b) - (r - r_cut) * gaussian_derivative(r_cut, r_cut ** 2, a, b)


class PrismGaussianAperture(HasEnergy, PrismCoefficients):

    def __init__(self, kx, ky, amplitude=None, radius=None, energy=None, save_tensor=True):
        self._radius = radius
        self._amplitude = amplitude

        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy)

    radius = notifying_property('_radius')
    amplitude = notifying_property('_amplitude')

    def _calculate_tensor(self):
        alpha_x = self.kx * self.wavelength
        alpha_y = self.ky * self.wavelength

        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)
        alpha_cut = tf.reduce_max(alpha)

        amplitude = tf.reshape(tf.constant(self._amplitude, dtype=tf.float32), (1, -1))
        radius = tf.reshape(tf.constant(self._radius, dtype=tf.float32), (1, -1))

        # print(gaussian(alpha2[:, None], amplitude, radius))
        # tf.reduce_sum(soft_gaussian(alpha[:, None], alpha2[:, None], amplitude, radius, alpha_cut), axis=1)
        return tf.reduce_sum(gaussian(alpha2[:, None], amplitude, radius), axis=1)


class PrismAperture(HasEnergy, PrismCoefficients):

    def __init__(self, kx, ky, radius=np.inf, rolloff=0., energy=None, save_tensor=True):

        self._radius = radius
        self._rolloff = rolloff

        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy)

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


class PrismAberration(HasEnergy, PrismCoefficients):

    def __init__(self, kx, ky, energy=None, save_tensor=True, parametrization='polar', **kwargs):

        if parametrization.lower() == 'polar':
            self._parametrization = PolarAberrations(**kwargs)

        else:
            raise RuntimeError()

        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy)

        self._parametrization.register_observer(self)

    @property
    def parametrization(self):
        return self._parametrization

    def set_parameters(self, parameters):
        self.notify_observers({'change': True})

        self.parametrization.set_parameters(parameters)

    def _calculate_tensor(self):

        alpha_x = self.kx * self.wavelength
        alpha_y = self.ky * self.wavelength
        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)
        phi = tf.atan2(alpha_x, alpha_y)
        tensor = self.parametrization(alpha=alpha, alpha2=alpha2, phi=phi)

        return complex_exponential(- 2 * np.pi / self.wavelength * tensor)


class PrismTranslate(PrismCoefficients):

    def __init__(self, kx, ky, position=None, save_tensor=True):

        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)

        if position is None:
            position = (0., 0.)

        self._position = self._validate_position(position)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        old = self.position
        self._position = self._validate_position(position)
        change = np.any(self._position != old)
        self.notify_observers({'name': '_position', 'old': old, 'new': position, 'change': change})

    def _validate_position(self, position):
        if isinstance(position, (np.ndarray, list, tuple)):
            position = np.array(position, dtype=np.float32)
            if position.shape != (2,):
                raise RuntimeError()

            return position

        else:
            raise RuntimeError('')

    def _calculate_tensor(self):

        tensor = complex_exponential(-2 * np.pi * (self.kx * self.position[0] + self.ky * self.position[1]))

        return tf.cast(tensor, tf.complex64)
