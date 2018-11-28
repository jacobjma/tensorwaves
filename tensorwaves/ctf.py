from collections import OrderedDict
from math import pi

import tensorflow as tf
from tensorwaves.bases import TensorFactory
from tensorwaves.utils import complex_exponential


class Aperture(TensorFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, radius=.1, rolloff=0.):
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy)

        self.radius = radius
        self.rolloff = rolloff

    def get_tensor(self):
        wavelength = self.accelerator.wavelength
        kx, ky = self.grid.fftfreq()

        alpha_x = kx * wavelength
        alpha_y = ky * wavelength

        alpha2 = alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2
        alpha = tf.sqrt(alpha2)

        if self.rolloff > 0.:
            tensor = .5 * (1 + tf.cos(pi * (alpha - self.radius) / self.rolloff))
            tensor *= tf.cast(alpha < self.radius + self.rolloff, tf.float32)
            tensor = tf.where(alpha > self.radius, tensor, tf.ones((self.grid.gpts, self.grid.gpts)))
        else:
            tensor = tf.cast(alpha < self.radius, tf.float32)

        return tf.cast(tensor[None, :, :], tf.complex64)


def link_property(symbol):
    def getter(self):
        return getattr(self, symbol)

    def setter(self, value):
        setattr(self, symbol, value)

    return property(getter, setter)


class ParameterizedCTF(object):

    def __init__(self, symbols, aliases):

        for symbol in symbols:
            setattr(ParameterizedCTF, symbol, 0.)

        for alias, symbol in zip(aliases, symbols):
            if alias is not None:
                setattr(ParameterizedCTF, alias, link_property(symbol))


class PolarCTF(ParameterizedCTF):

    def __init__(self):
        symbols = ('C10', 'C12', 'phi12',
                   'C21', 'phi21', 'C23', 'phi23',
                   'C30', 'C32', 'phi32', 'C34', 'phi34',
                   'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                   'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

        aliases = ('defocus', 'astig_mag', 'astig_angle',
                   'coma', 'coma_angle', 'astig_mag_2', 'astig_angle_2',
                   'Cs', None, None, None, None,
                   None, None, None, None, None, None,
                   'C5', None, None, None, None, None, None)

        ParameterizedCTF.__init__(self, symbols, aliases)

    def get_function(self):
        if any([getattr(self, symbol) != 0. for symbol in ('C10', 'C12', 'phi12')]):
            chi = lambda a, a2, b: (1 / 2. * a2 *
                                    (self.C10 +
                                     self.C12 * tf.cos(2. * (b - self.phi12))))
        if any([getattr(self, symbol) != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
            chi_old1 = chi
            chi = lambda a, a2, b: (chi_old1(a, a2, b) + 1 / 3. * a2 * a *
                                    (self.C21 * tf.cos(b - self.phi21) +
                                     self.C23 * tf.cos(3. * (b - self.phi23))))
        if any([getattr(self, symbol) != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
            chi_old2 = chi
            chi = lambda a, a2, b: (chi_old2(a, a2, b) + 1 / 4. * a2 ** 2 *
                                    (self.C30 +
                                     self.C32 * tf.cos(2. * (b - self.phi32)) +
                                     self.C34 * tf.cos(4. * (b - self.phi34))))
        if any([getattr(self, symbol) != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
            chi_old3 = chi
            chi = lambda a, a2, b: (chi_old3(a, a2, b) + 1 / 5. * a2 ** 2 * a *
                                    (self.C41 * tf.cos((b - self.phi41)) +
                                     self.C43 * tf.cos(3. * (b - self.phi43)) +
                                     self.C45 * tf.cos(5. * (b - self.phi41))))
        if any([getattr(self, symbol) != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
            chi_old4 = chi
            chi = lambda a, a2, b: (chi_old4(a, a2, b) + 1 / 6. * a2 ** 3 *
                                    (self.C50 +
                                     self.C52 * tf.cos(2. * (b - self.phi52)) +
                                     self.C54 * tf.cos(4. * (b - self.phi54)) +
                                     self.C56 * tf.cos(6. * (b - self.phi56))))
        return chi


class SymmetricCTF(ParameterizedCTF):

    def __init__(self):
        symbols = ('C10', 'C30', 'C50')
        aliases = ('defocus', 'Cs', 'C5')

        ParameterizedCTF.__init__(self, symbols, aliases)

    def get_function(self):
        if self.C10 != 0.:
            chi = lambda a, a2: 1 / 2. * a2 * p['C10']
        if self.C30 != 0.:
            chi_old1 = chi
            chi = lambda a, a2: chi_old1(a, a2) + 1 / 4. * a2 ** 2 * p['C30']
        if self.C50 != 0.:
            chi_old2 = chi
            chi = lambda a, a2: chi_old2(a, a2) + 1 / 6. * a2 ** 3 * p['C50']
        return chi


class CartesianCTF(ParameterizedCTF):

    def __init__(self):
        symbols = ('C10', 'C12a', 'C12b',
                   'C21a', 'C21b', 'C23a', 'C23b',
                   'C30', 'C32a', 'C32b', 'C34a', 'C34b')

        aliases = ('defocus', 'astig_x', 'astig_y',
                   'coma_x', 'coma_y', 'astig_x_2', 'astig_y_2',
                   'Cs', None, None, None, None)

        ParameterizedCTF.__init__(self, symbols, aliases)

    def get_function(self):

        # todo: implement 4th and 5th order
        if any([getattr(self, symbol) != 0. for symbol in ('C10', 'C12a', 'C12b')]):
            chi = lambda ax, ay, ax2, ay2, a2: (1 / 2. * (self.C10 * a2 +
                                                          self.C12a * (ax2 - ay2)) + self.C12b * ax * ay)
        if any([getattr(self, symbol) != 0. for symbol in ('C21a', 'C21b', 'C23a', 'C23b')]):
            chi_old1 = chi
            chi = (lambda ax, ay, ax2, ay2, a2: chi_old1(ax, ay, ax2, ay2, a2) +
                                                1 / 3. * (a2 * (self.C21a * ax + self.C21b * ay) +
                                                          self.C23a * ax * (ax2 - 3 * ay2) +
                                                          self.C23b * ay * (ay2 - 3 * ax2)))
        if any([getattr(self, symbol) != 0. for symbol in ('C30', 'C32a', 'C32b', 'C34a', 'C34b')]):
            chi_old2 = chi
            chi = (lambda ax, ay, ax2, ay2, a2: chi_old2(ax, ay, ax2, ay2, a2) +
                                                1 / 4. * (self.C30 * a2 ** 2 +
                                                          self.C32a * (ax2 ** 2 - ay2 ** 2) +
                                                          2 * self.C32b * ax * ay * a2 +
                                                          self.C34a * (ax2 ** 2 - 6 * ax2 * ay2 + ay2 ** 2) +
                                                          4 * self.C34b * (ax * ay2 * ay - ax2 * ax * ay)))
        return chi


class CTF(TensorFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, parametrization='polar', **kwargs):
        super().__init__(extent=extent, gpts=gpts, sampling=sampling, energy=energy)

        if parametrization.lower() == 'polar':
            self.parametrization = PolarCTF()
        elif parametrization.lower() == 'cartesian':
            raise NotImplementedError()
        else:
            raise RuntimeError()

        for symbol, value in kwargs.items():
            if not hasattr(self.parametrization, symbol):
                raise RuntimeError()

            setattr(self.parametrization, symbol, value)

    def _line_data(self, phi=0, kmax=1):
        wavelength = self.accelerator.wavelength

        k = tf.linspace(0., kmax, 1024)
        alpha = wavelength * k

        chi = 2 * pi / wavelength * self.parametrization.get_function()(alpha, alpha ** 2, phi)

        return k, complex_exponential(chi)

    def get_tensor(self):
        wavelength = self.accelerator.wavelength
        kx, ky = self.grid.fftfreq()

        alpha_x = kx * wavelength
        alpha_y = ky * wavelength

        alpha2 = alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2
        alpha = tf.sqrt(alpha2)
        alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)

        phi = tf.atan2(alpha_x, alpha_y)
        chi = self.parametrization.get_function()(alpha, alpha2, phi)

        return complex_exponential(2 * pi / wavelength * chi)[None, :, :]


def polar2cartesian(polar):
    cartesian = {}
    cartesian['C10'] = polar['C10']
    cartesian['C12a'] = - polar['C12'] * tf.cos(2 * polar['phi12'])
    cartesian['C12b'] = polar['C12'] * tf.cos(pi / 2 - 2 * polar['phi12'])
    cartesian['C21a'] = polar['C21'] * tf.cos(pi / 2 - polar['phi21'])
    cartesian['C21b'] = polar['C21'] * tf.cos(polar['phi21'])
    cartesian['C23a'] = polar['C23'] * tf.cos(3 * pi / 2. - 3 * polar['phi23'])
    cartesian['C23b'] = polar['C23'] * tf.cos(3 * polar['phi23'])
    cartesian['C30'] = polar['C30']
    cartesian['C32a'] = - polar['C32'] * tf.cos(2 * polar['phi32'])
    cartesian['C32b'] = polar['C32'] * tf.cos(pi / 2 - 2 * polar['phi32'])
    cartesian['C34a'] = polar['C34'] * tf.cos(-4 * polar['phi34'])
    K = tf.sqrt(3 + tf.sqrt(8.))
    cartesian['C34b'] = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K) * polar['C34'] * tf.cos(
        4 * tf.atan(1 / K) - 4 * polar['phi34'])
    return cartesian
