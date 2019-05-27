import numpy as np
import tensorflow as tf

from tensorwaves.bases import HasGridAndEnergy, FrequencyTransfer, notifying_property, PrismCoefficients, HasEnergy
from tensorwaves.bases import TensorWithGridAndEnergy
from tensorwaves.utils import squared_norm


class ApertureBase(HasEnergy, FrequencyTransfer):

    def __init__(self, cutoff=np.inf, rolloff=0., energy=None, save_tensor=True, energy_wrapper=None):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy, energy_wrapper=energy_wrapper)
        self._cutoff = cutoff
        self._rolloff = rolloff

    cutoff = notifying_property('_cutoff')
    rolloff = notifying_property('_rolloff')

    def _evaluate(self, alpha):
        if self.rolloff > 0.:
            tensor = .5 * (1 + tf.cos(np.pi * (alpha - self.cutoff) / self.rolloff))
            tensor *= tf.cast(alpha < (self.cutoff + self.rolloff), tf.float32)

            tensor = tf.where(alpha > self.cutoff, tensor, tf.ones(alpha.shape, dtype=tf.float32))

        else:
            tensor = tf.cast(alpha < self.cutoff, tf.float32)

        return tensor

    def line_profile(self, alpha_max=2, n=1024):
        alpha = tf.linspace(0., alpha_max, n)
        tensor = self._evaluate(alpha)
        return alpha, tensor


class Aperture(HasGridAndEnergy, ApertureBase):
    def __init__(self, cutoff=np.inf, rolloff=0., extent=None, gpts=None, sampling=None, energy=None, save_tensor=True,
                 grid=None, energy_wrapper=None):
        ApertureBase.__init__(self, cutoff=cutoff, rolloff=rolloff, save_tensor=save_tensor)
        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, grid=grid,
                                  energy_wrapper=energy_wrapper)

    def _calculate_tensor(self, *args):
        if not args:
            args = (squared_norm(*self.semiangles()),)

        tensor = ApertureBase._evaluate(self, args[0])

        return TensorWithGridAndEnergy(tensor[None], extent=self.extent, space='fourier')


class PrismAperture(PrismCoefficients, ApertureBase):
    def __init__(self, cutoff=np.inf, rolloff=0., kx=None, ky=None, energy=None, save_tensor=True, energy_wrapper=None):
        ApertureBase.__init__(self, cutoff=cutoff, rolloff=rolloff, energy=energy, save_tensor=save_tensor,
                              energy_wrapper=energy_wrapper)
        PrismCoefficients.__init__(self, kx=kx, ky=ky)

    def _calculate_tensor(self, *args):
        if not args:
            args = (tf.sqrt(self.kx ** 2 + self.ky ** 2) * self.wavelength,)

        tensor = ApertureBase._evaluate(self, args[0])
        return tensor

    def copy(self):
        return self.__class__(cutoff=self.cutoff, rolloff=self.rolloff, kx=self.kx, ky=self.ky,
                              save_tensor=self.save_tensor, energy_wrapper=self.energy_wrapper.copy())


def gaussian(r2, a, b):
    return a * tf.exp(-r2 / (b ** 2))


def gaussian_derivative(r, r2, a, b):
    return - 2 * a * 1 / b ** 2 * r * tf.exp(-r2 / b ** 2)


def soft_gaussian(r, r2, a, b, rc):
    y = gaussian(r2, a, b)
    c1 = - 2 * gaussian(rc ** 2, a, b) / rc ** 3 + gaussian_derivative(rc, rc ** 2, a, b) / rc ** 2
    c2 = - gaussian_derivative(rc, rc ** 2, a, b) / rc + 3 * gaussian(rc ** 2, a, b) / rc ** 2
    corr = c1 * r2 * r + c2 * r2
    return y - corr


# def soft_gaussian(r, r2, a, b, r_cut):
#     return (gaussian(r2, a, b) - gaussian(r_cut ** 2, a, b) - (r - r_cut) *
#             gaussian_derivative(r_cut, r_cut ** 2, a, b))


class GaussianEnvelopeBase(HasEnergy, FrequencyTransfer):

    def __init__(self, width=np.inf, scale=1., energy=None, save_tensor=True, energy_wrapper=None):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy, energy_wrapper=energy_wrapper)
        self._scale = scale
        self._width = width

    scale = notifying_property('_scale')
    width = notifying_property('_width')

    def _evaluate(self, alpha2):
        scale = tf.cast(tf.reshape(self._scale, (-1, 1, 1)), tf.float32)
        width = tf.cast(tf.reshape(self._width, (-1, 1, 1)), tf.float32)

        # tensor = scale * tf.exp(-alpha2 / width ** 2)
        tensor = gaussian(alpha2, scale, width)

        #tensor = soft_gaussian(tf.sqrt(alpha2), alpha2, scale, width, .03)

        tensor = tf.reduce_sum(tensor, axis=0)
        return tensor

    def line_profile(self, alpha_max=2, n=1024):
        alpha = tf.linspace(0., alpha_max, n)
        tensor = self._calculate_tensor(alpha ** 2)
        return alpha, tensor


class GaussianEnvelope(HasGridAndEnergy, GaussianEnvelopeBase):
    def __init__(self, width=np.inf, scale=1., extent=None, gpts=None, sampling=None, energy=None,
                 save_tensor=True, grid=None, energy_wrapper=None):
        GaussianEnvelopeBase.__init__(self, width=width, scale=scale, save_tensor=save_tensor)
        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, grid=grid,
                                  energy_wrapper=energy_wrapper)

    def _calculate_tensor(self, *args):
        if not args:
            args = (squared_norm(*self.semiangles()),)

        tensor = GaussianEnvelopeBase._evaluate(self, args[0])

        return TensorWithGridAndEnergy(tensor[None], extent=self.extent, space='fourier')


class PrismGaussianEnvelope(PrismCoefficients, GaussianEnvelope):
    def __init__(self, width=np.inf, scale=1., kx=None, ky=None, energy=None, save_tensor=True, energy_wrapper=None):
        GaussianEnvelopeBase.__init__(self, width=width, scale=scale, energy=energy, save_tensor=save_tensor,
                                      energy_wrapper=energy_wrapper)
        PrismCoefficients.__init__(self, kx=kx, ky=ky)

    def _calculate_tensor(self, *args):
        if not args:
            args = ((self.kx ** 2 + self.ky ** 2) * self.wavelength ** 2,)
        tensor = GaussianEnvelopeBase._evaluate(self, args[0])
        return tensor[0]

    def copy(self):
        return self.__class__(width=self.width, scale=self.scale, kx=self.kx, ky=self.ky,
                              save_tensor=self.save_tensor, energy_wrapper=self.energy_wrapper.copy())


class TemporalEnvelopeBase(HasEnergy, FrequencyTransfer):

    def __init__(self, focal_spread=0., energy=None, save_tensor=True, energy_wrapper=None):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy, energy_wrapper=energy_wrapper)

        self._focal_spread = focal_spread

    focal_spread = notifying_property('_focal_spread')

    def _evaluate(self, alpha2):
        tensor = tf.exp(- (.5 * np.pi / self.wavelength * self.focal_spread * alpha2) ** 2)

        return tensor

    def line_profile(self, alpha_max=0.02, n=1024):
        alpha = tf.linspace(0., alpha_max, n)
        tensor = self._evaluate(alpha ** 2)
        return alpha, tensor


class TemporalEnvelope(HasGridAndEnergy, TemporalEnvelopeBase):
    def __init__(self, focal_spread=0., extent=None, gpts=None, sampling=None, energy=None,
                 save_tensor=True, grid=None, energy_wrapper=None):
        TemporalEnvelopeBase.__init__(self, focal_spread=focal_spread, save_tensor=save_tensor)
        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, grid=grid,
                                  energy_wrapper=energy_wrapper)

    def _calculate_tensor(self, *args):
        if not args:
            args = (squared_norm(*self.semiangles()),)

        tensor = TemporalEnvelopeBase._evaluate(self, args[0])
        return TensorWithGridAndEnergy(tensor[None], extent=self.extent, space='fourier')


class PrismTemporalEnvelope(PrismCoefficients, TemporalEnvelopeBase):

    def __init__(self, focal_spread=0., kx=None, ky=None, energy=None, save_tensor=True, energy_wrapper=None):
        TemporalEnvelopeBase.__init__(self, focal_spread=focal_spread, energy=energy, save_tensor=save_tensor,
                                      energy_wrapper=energy_wrapper)
        PrismCoefficients.__init__(self, kx=kx, ky=ky)

    def _calculate_tensor(self, *args):
        if not args:
            args = ((self.kx ** 2 + self.ky ** 2) * self.wavelength ** 2,)

        tensor = TemporalEnvelopeBase._evaluate(self, args[0])
        return tensor

    def copy(self):
        return self.__class__(focal_spread=self.focal_spread, kx=self.kx, ky=self.ky,
                              save_tensor=self.save_tensor, energy_wrapper=self.energy_wrapper.copy())
