from collections import defaultdict

import numpy as np
import tensorflow as tf

from tensorwaves.bases import FrequencyTransfer, HasGridAndEnergy, TensorWithGridAndEnergy, HasEnergy, PrismCoefficients
from tensorwaves.utils import complex_exponential, squared_norm, angle


def parametrization_property(key):
    def getter(self):
        return self._parameters[key]

    def setter(self, value):
        old = getattr(self, key)
        self._parameters[key] = value
        change = old != value
        self.notify_observers({'name': key, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


class Aberrations(HasEnergy, FrequencyTransfer):

    def __init__(self, symbols, aliases, parameters, energy=None, save_tensor=True, energy_wrapper=None):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy, energy_wrapper=energy_wrapper)

        self._symbols = symbols
        self._aliases = aliases
        self._parameters = dict(zip(symbols, [0.] * len(symbols)))
        self.set_parameters(parameters)

        self.observe(self)
        self.observe(self.energy_wrapper)

    def _calculate_chi(self, *args):
        raise NotImplementedError()

    @property
    def parameters(self):
        return self._parameters

    def set_parameters(self, parameters):
        for symbol, value in parameters.items():
            if symbol in self._parameters.keys():
                self._parameters[symbol] = np.float32(value)

            elif symbol in self._aliases.keys():
                self._parameters[self._aliases[symbol]] = np.float32(value)

            else:
                raise RuntimeError('{}'.format(symbol))

    def line_profile(self, *args):
        raise NotImplementedError()


class PolarAberrationsBase(Aberrations):

    def __init__(self, parameters=None, energy=None, save_tensor=True, energy_wrapper=None, **kwargs):

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)

        symbols = ('C10', 'C12', 'phi12',
                   'C21', 'phi21', 'C23', 'phi23',
                   'C30', 'C32', 'phi32', 'C34', 'phi34',
                   'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                   'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

        aliases = {'defocus': 'C10', 'astigmatism': 'C12', 'astigmatism_angle': 'phi12',
                   'coma': 'C21', 'coma_angle': 'phi21',
                   'Cs': 'C30',
                   'C5': 'C50'}

        Aberrations.__init__(self, symbols, aliases, parameters, energy=energy, save_tensor=save_tensor,
                             energy_wrapper=energy_wrapper)

    C10 = parametrization_property('C10')
    C12 = parametrization_property('C12')
    phi12 = parametrization_property('phi12')

    C21 = parametrization_property('C21')
    phi21 = parametrization_property('phi21')
    C23 = parametrization_property('C23')
    phi23 = parametrization_property('phi23')

    C30 = parametrization_property('C30')
    C32 = parametrization_property('C32')
    phi32 = parametrization_property('phi32')
    C34 = parametrization_property('C34')
    phi34 = parametrization_property('phi34')

    C41 = parametrization_property('C41')
    phi41 = parametrization_property('phi41')
    C43 = parametrization_property('C43')
    phi43 = parametrization_property('phi43')
    C45 = parametrization_property('C45')
    phi45 = parametrization_property('phi45')

    C50 = parametrization_property('C50')
    C52 = parametrization_property('C52')
    phi52 = parametrization_property('phi52')
    C54 = parametrization_property('C54')
    phi54 = parametrization_property('phi54')
    C56 = parametrization_property('C56')
    phi56 = parametrization_property('phi56')

    defocus = parametrization_property('C10')
    astigmatism = parametrization_property('C12')
    astigmatism_angle = parametrization_property('phi12')
    coma = parametrization_property('C21')
    coma_angle = parametrization_property('phi21')
    Cs = parametrization_property('C30')
    C5 = parametrization_property('C50')

    def _evaluate(self, *args):
        tensor = complex_exponential(2 * np.pi / self.wavelength * self._calculate_chi(*args))
        return tensor

    def _calculate_chi(self, alpha, alpha2, phi):
        tensor = tf.zeros(alpha.shape)

        if any([getattr(self, symbol) != 0. for symbol in ('C10', 'C12', 'phi12')]):
            tensor += (1 / 2. * alpha2 *
                       (self.C10 +
                        self.C12 * tf.cos(2. * (phi - self.phi12))))

        if any([getattr(self, symbol) != 0. for symbol in ('C21', 'phi21', 'C23', 'phi23')]):
            tensor += (1 / 3. * alpha2 * alpha *
                       (self.C21 * tf.cos(phi - self.phi21) +
                        self.C23 * tf.cos(3. * (phi - self.phi23))))

        if any([getattr(self, symbol) != 0. for symbol in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
            tensor += (1 / 4. * alpha2 ** 2 *
                       (self.C30 +
                        self.C32 * tf.cos(2. * (phi - self.phi32)) +
                        self.C34 * tf.cos(4. * (phi - self.phi34))))

        if any([getattr(self, symbol) != 0. for symbol in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
            tensor += (1 / 5. * alpha2 ** 2 * alpha *
                       (self.C41 * tf.cos((phi - self.phi41)) +
                        self.C43 * tf.cos(3. * (phi - self.phi43)) +
                        self.C45 * tf.cos(5. * (phi - self.phi45))))

        if any([getattr(self, symbol) != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
            tensor += (1 / 6. * alpha2 ** 3 *
                       (self.C50 +
                        self.C52 * tf.cos(2. * (phi - self.phi52)) +
                        self.C54 * tf.cos(4. * (phi - self.phi54)) +
                        self.C56 * tf.cos(6. * (phi - self.phi56))))

        return tensor

    def line_profile(self, phi=0., alpha_max=.02, n=1024):
        alpha = tf.linspace(0., alpha_max, n)
        return alpha, self._evaluate(alpha, alpha ** 2, phi)


class PolarAberrations(HasGridAndEnergy, PolarAberrationsBase):

    def __init__(self, parameters=None, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True,
                 grid=None, energy_wrapper=None, **kwargs):
        PolarAberrationsBase.__init__(self, parameters=parameters, save_tensor=save_tensor, **kwargs)

        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, grid=grid,
                                  energy_wrapper=energy_wrapper)

    def _calculate_tensor(self, *args):
        if not args:
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha = tf.sqrt(alpha2)
            phi = angle(alpha_x, alpha_y)
            args = (alpha, alpha2, phi)

        tensor = PolarAberrationsBase._evaluate(self, *args)

        return TensorWithGridAndEnergy(tensor[None], extent=self.extent, energy=self.energy, space='fourier')


class PrismPolarAberrations(PrismCoefficients, PolarAberrationsBase):

    def __init__(self, parameters=None, kx=None, ky=None, energy=None, save_tensor=True, energy_wrapper=None, **kwargs):
        PolarAberrationsBase.__init__(self, parameters=parameters, energy=energy, save_tensor=save_tensor,
                                      energy_wrapper=energy_wrapper, **kwargs)
        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)

    def _calculate_tensor(self, *args):
        if not args:
            alpha_x, alpha_y = self.kx * self.wavelength, self.ky * self.wavelength
            alpha2 = alpha_x ** 2 + alpha_y ** 2
            alpha = tf.sqrt(alpha2)
            phi = tf.atan2(alpha_x, alpha_y)
            args = (alpha, alpha2, phi)

        tensor = PolarAberrationsBase._evaluate(self, *args)

        return tensor

    def copy(self):
        kwargs = {symbol: getattr(self, symbol) for symbol in self._symbols}
        new = self.__class__(kx=self.kx, ky=self.ky, save_tensor=self.save_tensor,
                             energy_wrapper=self.energy_wrapper.copy(), **kwargs)

        return new


class CartesianAberrations(Aberrations):
    pass


class PrismCartesianAberrations(Aberrations):
    pass

#     def __init__(self, parameters=None, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True, grid=None,
#                  energy_wrapper=None, **kwargs):
#
#         if parameters is None:
#             parameters = {}
#
#         parameters.update(kwargs)
#
#         symbols = ('C10', 'C12a', 'C12b',
#                    'C21a', 'C21b', 'C23a', 'C23b',
#                    'C30', 'C32a', 'C32b', 'C34a', 'C34b')
#
#         aliases = {'defocus': 'C10', 'astigmatism_x': 'C12a', 'astigmatism_y': 'C12b',
#                    'coma': 'C21', 'coma_angle': 'phi21',
#                    'C30': 'Cs',
#                    'C50': 'C5'}
#
#         Aberrations.__init__(self, symbols, aliases, parameters, extent=extent, gpts=gpts, sampling=sampling,
#                              energy=energy, save_tensor=save_tensor, grid=grid, energy_wrapper=energy_wrapper)
#
#     C10 = parametrization_property('C10')
#     C12a = parametrization_property('C12a')
#     C12b = parametrization_property('C12b')
#
#     C21a = parametrization_property('C21a')
#     C21b = parametrization_property('C21b')
#     C23a = parametrization_property('C23a')
#     C23b = parametrization_property('C23b')
#
#     C30 = parametrization_property('C30')
#     C32a = parametrization_property('C32a')
#     C32b = parametrization_property('C32b')
#     C34a = parametrization_property('C34a')
#     C34b = parametrization_property('C34b')
#
#     defocus = parametrization_property('C10')
#     astigmatism_x = parametrization_property('C12a')
#     astigmatism_y = parametrization_property('C12b')
#     coma_x = parametrization_property('C21a')
#     coma_y = parametrization_property('C21b')
#
#     def line_profile(self, phi=0., k_max=2, n=1024):
#         raise NotImplementedError()
#
#     def _calculate_tensor(self, *args):
#         if not args:
#             alpha_x, alpha_y = self.semiangles()
#             alpha2 = squared_norm(alpha_x, alpha_y)
#             alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)
#             alpha_x_2 = alpha_x ** 2
#             alpha_y_2 = alpha_y ** 2
#             args = (alpha_x, alpha_y, alpha_x_2, alpha_y_2, alpha2)
#
#         tensor = complex_exponential(2 * np.pi / self.wavelength * self._calculate_chi(*args))[None]
#
#         return TensorWithGridAndEnergy(tensor, extent=self.extent, energy=self.energy, space='fourier')
#
#     def _calculate_chi(self, ax, ay, ax2, ay2, a2):
#         tensor = tf.zeros(ax.shape)
#
#         if any([getattr(self, symbol) != 0. for symbol in ('C10', 'C12a', 'C12b')]):
#             tensor += (1 / 2. * (self.C10 * a2 +
#                                  self.C12a * (ax2 - ay2)) + self.C12b * ax * ay)
#
#         if any([getattr(self, symbol) != 0. for symbol in ('C21a', 'C21b', 'C23a', 'C23b')]):
#             tensor += 1 / 3. * (a2 * (self.C21a * ax + self.C21b * ay) +
#                                 self.C23a * ax * (ax2 - 3 * ay2) +
#                                 self.C23b * ay * (ay2 - 3 * ax2))
#
#         if any([getattr(self, symbol) != 0. for symbol in ('C30', 'C32a', 'C32b', 'C34a', 'C34b')]):
#             tensor += 1 / 4. * (self.C30 * a2 ** 2 +
#                                 self.C32a * (ax2 ** 2 - ay2 ** 2) +
#                                 2 * self.C32b * ax * ay * a2 +
#                                 self.C34a * (ax2 ** 2 - 6 * ax2 * ay2 + ay2 ** 2) +
#                                 4 * self.C34b * (ax * ay2 * ay - ax2 * ax * ay))
#
#         return tensor
#
#
# def polar2cartesian(polar):
#     polar = defaultdict(lambda: 0, polar)
#
#     cartesian = {}
#     cartesian['C10'] = polar['C10']
#     cartesian['C12a'] = - polar['C12'] * np.cos(2 * polar['phi12'])
#     cartesian['C12b'] = polar['C12'] * np.sin(2 * polar['phi12'])
#     cartesian['C21a'] = polar['C21'] * np.sin(polar['phi21'])
#     cartesian['C21b'] = polar['C21'] * np.cos(polar['phi21'])
#     cartesian['C23a'] = - polar['C23'] * np.sin(3 * polar['phi23'])
#     cartesian['C23b'] = polar['C23'] * np.cos(3 * polar['phi23'])
#     cartesian['C30'] = polar['C30']
#     cartesian['C32a'] = - polar['C32'] * np.cos(2 * polar['phi32'])
#     cartesian['C32b'] = polar['C32'] * np.cos(np.pi / 2 - 2 * polar['phi32'])
#     cartesian['C34a'] = polar['C34'] * np.cos(-4 * polar['phi34'])
#     K = np.sqrt(3 + np.sqrt(8.))
#     cartesian['C34b'] = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K) * polar['C34'] * np.cos(
#         4 * np.arctan(1 / K) - 4 * polar['phi34'])
#
#     return cartesian
#
#
# def cartesian2polar(cartesian):
#     cartesian = defaultdict(lambda: 0, cartesian)
#
#     polar = {}
#     polar['C10'] = cartesian['C10']
#     polar['C12'] = - np.sqrt(cartesian['C12a'] ** 2 + cartesian['C12b'] ** 2)
#     polar['phi12'] = - np.arctan2(cartesian['C12b'], cartesian['C12a']) / 2.
#     polar['C21'] = np.sqrt(cartesian['C21a'] ** 2 + cartesian['C21b'] ** 2)
#     polar['phi21'] = np.arctan2(cartesian['C21a'], cartesian['C21b'])
#     polar['C23'] = np.sqrt(cartesian['C23a'] ** 2 + cartesian['C23b'] ** 2)
#     polar['phi23'] = -np.arctan2(cartesian['C23a'], cartesian['C23b']) / 3.
#     polar['C30'] = cartesian['C30']
#     polar['C32'] = -np.sqrt(cartesian['C32a'] ** 2 + cartesian['C32b'] ** 2)
#     polar['phi32'] = -np.arctan2(cartesian['C32b'], cartesian['C32a']) / 2.
#     polar['C34'] = np.sqrt(cartesian['C34a'] ** 2 + cartesian['C34b'] ** 2)
#     polar['phi34'] = np.arctan2(cartesian['C34b'], cartesian['C34a']) / 4
#
#     return polar
