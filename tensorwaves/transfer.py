from collections import defaultdict

import numpy as np
import tensorflow as tf

from tensorwaves.bases import TensorWithEnergy, Tensor, notifying_property, Observable, HasGrid, HasEnergy, \
    HasGridAndEnergy, TensorFactory, Grid, EnergyProperty
from tensorwaves.utils import complex_exponential


def squared_norm(x, y):
    return x[:, None] ** 2 + y[None, :] ** 2


def angle(x, y):
    return tf.atan2(x[:, None], y[None, :])


class FrequencyTransfer(HasGrid, HasEnergy, HasGridAndEnergy, TensorFactory, Observable):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        HasEnergy.__init__(self, energy=energy)
        HasGridAndEnergy.__init__(self)
        TensorFactory.__init__(self, save_tensor=save_tensor)
        Observable.__init__(self)

        self.observe(self)
        self.observe(self._grid)
        self.observe(self._energy)

    def apply(self, waves):
        return waves.apply_frequency_transfer(self)


class Aperture(FrequencyTransfer):

    def __init__(self, radius=np.inf, rolloff=0., extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        FrequencyTransfer.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                   save_tensor=save_tensor)

        self._radius = radius
        self._rolloff = rolloff

    radius = notifying_property('_radius')
    rolloff = notifying_property('_rolloff')

    def _calculate_tensor(self, alpha=None):
        if alpha is None:
            alpha = tf.sqrt(squared_norm(*self.semiangles()))

        if self.rolloff > 0.:
            tensor = .5 * (1 + tf.cos(np.pi * (alpha - self.radius) / self.rolloff))
            tensor *= tf.cast(alpha < (self.radius + self.rolloff), tf.float32)
            tensor = tf.where(alpha > self.radius, tensor, tf.ones(alpha.shape))
        else:
            tensor = tf.cast(alpha < self.radius, tf.float32)

        return TensorWithEnergy(tf.cast(tensor[None, :, :], tf.complex64), extent=self.extent, space='fourier',
                                energy=self.energy)

    # def copy(self):
    #     self.__class__(radius=self.radius, rolloff=self.rolloff, save_tensor=self._save_tensor, energy=self.energy)
    #
    #     return


class TemporalEnvelope(FrequencyTransfer):

    def __init__(self, focal_spread=0., extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):

        FrequencyTransfer.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                   save_tensor=save_tensor)

        self._focal_spread = np.float32(focal_spread)

    focal_spread = notifying_property('_focal_spread')

    def _function(self, alpha):
        return tf.exp(- (.5 * np.pi / self.wavelength * self.focal_spread * alpha ** 2) ** 2)

    def _calculate_tensor(self, alpha=None):
        if alpha is None:
            alpha = tf.sqrt(squared_norm(*self.semiangles()))

        if self.focal_spread > 0.:
            tensor = self._function(alpha)
        else:
            tensor = tf.ones(alpha.shape)

        return TensorWithEnergy(tf.cast(tensor[None, :, :], tf.complex64), extent=self.extent, space='fourier',
                                energy=self.energy)

    def line_data(self, k_max=2, n=1024):
        k = tf.linspace(0., k_max, n)
        alpha = self.wavelength * k
        tensor = self._function(alpha)

        return k, tensor


class Parametrization(Observable):

    def __init__(self, symbols, aliases, parameters):
        Observable.__init__(self)

        self._aliases = aliases
        self._parameters = dict(zip(symbols, [0.] * len(symbols)))
        self.set_parameters(parameters)

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


def parametrization_property(key):
    def getter(self):
        return self._parameters[key]

    def setter(self, value):
        old = getattr(self, key)
        self._parameters[key] = value
        change = old != value
        self.notify_observers({'name': key, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


class PolarAberrations(Parametrization):

    def __init__(self, parameters=None, **kwargs):

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

        Parametrization.__init__(self, symbols, aliases, parameters)

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

    def to_zernike(self, aperture_radius):
        parameters = polar2zernike(self.parameters, aperture_radius)

        max_order = 0
        for key in parameters.keys():
            max_order = max(max_order, key[0])

        return ZernikeAberrations(aperture_radius=aperture_radius, parameters=parameters, max_order=max_order)

    def __call__(self, alpha, alpha2, phi):
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


class CartesianAberrations(Parametrization):

    def __init__(self, parameters=None, **kwargs):

        if parameters is None:
            parameters = {}

        parameters.update(kwargs)

        symbols = ('C10', 'C12a', 'C12b',
                   'C21a', 'C21b', 'C23a', 'C23b',
                   'C30', 'C32a', 'C32b', 'C34a', 'C34b')

        aliases = {'defocus': 'C10', 'astigmatism_x': 'C12a', 'astigmatism_y': 'C12b',
                   'coma': 'C21', 'coma_angle': 'phi21',
                   'C30': 'Cs',
                   'C50': 'C5'}

        Parametrization.__init__(self, symbols, aliases, kwargs)

    C10 = parametrization_property('C10')
    C12a = parametrization_property('C12a')
    C12b = parametrization_property('C12b')

    C21a = parametrization_property('C21a')
    C21b = parametrization_property('C21b')
    C23a = parametrization_property('C23a')
    C23b = parametrization_property('C23b')

    C30 = parametrization_property('C30')
    C32a = parametrization_property('C32a')
    C32b = parametrization_property('C32b')
    C34a = parametrization_property('C34a')
    C34b = parametrization_property('C34b')

    defocus = parametrization_property('C10')
    astigmatism_x = parametrization_property('C12a')
    astigmatism_y = parametrization_property('C12b')
    coma_x = parametrization_property('C21a')
    coma_y = parametrization_property('C21b')

    def __call__(self, ax, ay, ax2, ay2, a2):
        tensor = tf.zeros(ax.shape)

        if any([getattr(self, symbol) != 0. for symbol in ('C10', 'C12a', 'C12b')]):
            tensor += (1 / 2. * (self.C10 * a2 +
                                 self.C12a * (ax2 - ay2)) + self.C12b * ax * ay)

        if any([getattr(self, symbol) != 0. for symbol in ('C21a', 'C21b', 'C23a', 'C23b')]):
            tensor += 1 / 3. * (a2 * (self.C21a * ax + self.C21b * ay) +
                                self.C23a * ax * (ax2 - 3 * ay2) +
                                self.C23b * ay * (ay2 - 3 * ax2))

        if any([getattr(self, symbol) != 0. for symbol in ('C30', 'C32a', 'C32b', 'C34a', 'C34b')]):
            tensor += 1 / 4. * (self.C30 * a2 ** 2 +
                                self.C32a * (ax2 ** 2 - ay2 ** 2) +
                                2 * self.C32b * ax * ay * a2 +
                                self.C34a * (ax2 ** 2 - 6 * ax2 * ay2 + ay2 ** 2) +
                                4 * self.C34b * (ax * ay2 * ay - ax2 * ax * ay))

        return tensor


def polar2cartesian(polar):
    polar = defaultdict(lambda: 0, polar)

    cartesian = {}
    cartesian['C10'] = polar['C10']
    cartesian['C12a'] = - polar['C12'] * np.cos(2 * polar['phi12'])
    cartesian['C12b'] = polar['C12'] * np.sin(2 * polar['phi12'])
    cartesian['C21a'] = polar['C21'] * np.sin(polar['phi21'])
    cartesian['C21b'] = polar['C21'] * np.cos(polar['phi21'])
    cartesian['C23a'] = - polar['C23'] * np.sin(3 * polar['phi23'])
    cartesian['C23b'] = polar['C23'] * np.cos(3 * polar['phi23'])
    cartesian['C30'] = polar['C30']
    cartesian['C32a'] = - polar['C32'] * np.cos(2 * polar['phi32'])
    cartesian['C32b'] = polar['C32'] * np.cos(np.pi / 2 - 2 * polar['phi32'])
    cartesian['C34a'] = polar['C34'] * np.cos(-4 * polar['phi34'])
    K = np.sqrt(3 + np.sqrt(8.))
    cartesian['C34b'] = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K) * polar['C34'] * np.cos(
        4 * np.arctan(1 / K) - 4 * polar['phi34'])

    return cartesian  # {key: value for key, value in cartesian.items() if value != 0.}


def cartesian2polar(cartesian):
    cartesian = defaultdict(lambda: 0, cartesian)

    polar = {}
    polar['C10'] = cartesian['C10']
    polar['C12'] = - np.sqrt(cartesian['C12a'] ** 2 + cartesian['C12b'] ** 2)
    polar['phi12'] = - np.arctan2(cartesian['C12b'], cartesian['C12a']) / 2.
    polar['C21'] = np.sqrt(cartesian['C21a'] ** 2 + cartesian['C21b'] ** 2)
    polar['phi21'] = np.arctan2(cartesian['C21a'], cartesian['C21b'])
    polar['C23'] = np.sqrt(cartesian['C23a'] ** 2 + cartesian['C23b'] ** 2)
    polar['phi23'] = -np.arctan2(cartesian['C23a'], cartesian['C23b']) / 3.
    polar['C30'] = cartesian['C30']
    polar['C32'] = -np.sqrt(cartesian['C32a'] ** 2 + cartesian['C32b'] ** 2)
    polar['phi32'] = -np.arctan2(cartesian['C32b'], cartesian['C32a']) / 2.
    polar['C34'] = np.sqrt(cartesian['C34a'] ** 2 + cartesian['C34b'] ** 2)
    polar['phi34'] = np.arctan2(cartesian['C34b'], cartesian['C34a']) / 4

    return polar  # {key: value for key, value in polar.items() if value != 0.}


def zernike_polynomial(rho, phi, n, m):
    assert n >= m
    assert (n + m) % 2 == 0

    if m >= 0:
        even = True
    else:
        even = False

    def factorial(n):
        return np.prod(range(1, n + 1)).astype(int)

    def normalization(n, m, k):
        return (-1) ** k * factorial(n - k) // (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))

    m = abs(m)

    R = np.zeros_like(rho)
    for k in range(0, (n - m) // 2 + 1):
        if (n - 2 * k) > 1:
            R += normalization(n, m, k) * rho ** (n - 2 * k)

    if even:
        Z = R * np.cos(m * phi)
    else:
        Z = R * np.sin(m * phi)

    return Z


class ZernikeExpansion(object):

    def __init__(self, coefficients, basis, indices):
        self._coefficients = coefficients
        self._basis = basis
        self._indices = indices

    def to_parametrization(self):
        pass

    def sum(self):
        return tf.reduce_sum(self._basis * self._coefficients[:, None], axis=0)


class ZernikeAberrations(Parametrization):

    def __init__(self, aperture_radius, max_order=6, parameters=None):

        self._aperture_radius = aperture_radius

        if parameters is None:
            parameters = {}

        aliases = {}
        symbols = []

        symmetric = False

        for n in range(1, max_order + 1):
            for m in range(-n, n + 1):
                if (not symmetric) | (m == 0):
                    if (n - m) % 2 == 0:
                        symbols.append((n, m))

        Parametrization.__init__(self, symbols, aliases, parameters)

    @property
    def aperture_radius(self):
        return self._aperture_radius

    def expansion(self, k, phi):
        k /= self._aperture_radius

        indices = []
        expansion = []
        coefficients = []
        for (n, m), value in self.parameters.items():
            indices.append((n, m))
            expansion.append(zernike_polynomial(k, phi, n, m))
            coefficients.append(value)

        expansion = tf.convert_to_tensor(expansion)
        coefficients = tf.convert_to_tensor(coefficients)

        return indices, expansion, coefficients

    def to_polar(self):
        parameters = zernike2polar(self.parameters, self.aperture_radius)
        return PolarAberrations(parameters=parameters)

    def __call__(self, k, phi):
        k /= self._aperture_radius

        Z = tf.zeros(k.shape)
        for (n, m), value in self.parameters.items():
            Z += value * zernike_polynomial(k, phi, n, m)
        return Z


def polar_aberration_order(symbol):
    for letter in list(symbol):
        try:
            return int(letter)
        except:
            pass


def polar2zernike(polar, aperture_radius):
    polar = defaultdict(lambda: 0, polar)

    for symbol, value in polar.items():
        if symbol[0] == 'C':
            polar[symbol] *= aperture_radius ** (polar_aberration_order(symbol) + 1)

    zernike = {}
    zernike[(1, 1)] = 2 * polar['C21'] / 9. * np.cos(polar['phi21']) + polar['C41'] / 10. * np.cos(polar['phi41'])
    zernike[(1, -1)] = 2 * polar['C21'] / 9. * np.sin(polar['phi21']) + polar['C41'] / 10. * np.sin(polar['phi41'])

    zernike[(2, 0)] = polar['C10'] / 4. + polar['C30'] / 8. + 3 / 40. * polar['C50']
    zernike[(2, 2)] = polar['C12'] / 2. * np.cos(2 * polar['phi12']) + \
                      3 * polar['C32'] / 16. * np.cos(2 * polar['phi32']) + \
                      (1 / 6. - 1 / 15.) * polar['C52'] * np.cos(2 * polar['phi52'])
    zernike[(2, -2)] = polar['C12'] / 2. * np.sin(2 * polar['phi12']) + \
                       3 * polar['C32'] / 16. * np.sin(2 * polar['phi32']) + \
                       (1 / 6. - 1 / 15.) * polar['C52'] * np.sin(2 * polar['phi52'])

    zernike[(3, 1)] = polar['C21'] / 9. * np.cos(polar['phi21']) + 4 * polar['C41'] / 50. * np.cos(polar['phi41'])
    zernike[(3, -1)] = polar['C21'] / 9. * np.sin(polar['phi21']) + 4 * polar['C41'] / 50. * np.sin(polar['phi41'])
    zernike[(3, 3)] = polar['C23'] / 3. * np.cos(3 * polar['phi23']) + 4 * polar['C43'] / 25. * np.cos(
        3 * polar['phi43'])
    zernike[(3, -3)] = polar['C23'] / 3. * np.sin(3 * polar['phi23']) + 4 * polar['C43'] / 25. * np.sin(
        3 * polar['phi43'])

    zernike[(4, 0)] = polar['C30'] / 24. + polar['C50'] / 24.
    zernike[(4, 2)] = polar['C32'] / 16. * np.cos(2 * polar['phi32']) + polar['C52'] / 18. * np.cos(2 * polar['phi52'])
    zernike[(4, -2)] = polar['C32'] / 16. * np.sin(2 * polar['phi32']) + polar['C52'] / 18. * np.sin(2 * polar['phi52'])
    zernike[(4, 4)] = polar['C34'] / 4. * np.cos(4 * polar['phi34']) + 5 * polar['C54'] / 36. * np.cos(
        4 * polar['phi54'])
    zernike[(4, -4)] = polar['C34'] / 4. * np.sin(4 * polar['phi34']) + 5 * polar['C54'] / 36. * np.sin(
        4 * polar['phi54'])

    zernike[(5, 1)] = polar['C41'] / 50. * np.cos(polar['phi41'])
    zernike[(5, -1)] = polar['C41'] / 50. * np.sin(polar['phi41'])
    zernike[(5, 3)] = polar['C43'] / 25. * np.cos(3 * polar['phi43'])
    zernike[(5, -3)] = polar['C43'] / 25. * np.sin(3 * polar['phi43'])
    zernike[(5, 5)] = polar['C45'] / 5. * np.cos(5 * polar['phi45'])
    zernike[(5, -5)] = polar['C45'] / 5. * np.sin(5 * polar['phi45'])

    zernike[(6, 0)] = polar['C50'] / 120.
    zernike[(6, 2)] = polar['C52'] / 90. * np.cos(2 * polar['phi52'])
    zernike[(6, -2)] = polar['C52'] / 90. * np.sin(2 * polar['phi52'])
    zernike[(6, 4)] = polar['C54'] / 36. * np.cos(4 * polar['phi54'])
    zernike[(6, -4)] = polar['C54'] / 36. * np.sin(4 * polar['phi54'])
    zernike[(6, 6)] = polar['C56'] / 6. * np.cos(6 * polar['phi56'])
    zernike[(6, -6)] = polar['C56'] / 6. * np.sin(6 * polar['phi56'])

    return {key: value for key, value in zernike.items() if value != 0.}


def zernike2polar(zernike, aperture_radius):
    zernike = defaultdict(lambda: 0., zernike)

    polar = {}
    polar['C50'] = 120 * zernike[(6, 0)]
    polar['C52'] = np.sqrt(zernike[(6, -2)] ** 2 + zernike[(6, 2)] ** 2) * 90
    polar['phi52'] = np.arctan2(zernike[(6, -2)], zernike[(6, 2)]) / 2.
    polar['C54'] = np.sqrt(zernike[(6, -4)] ** 2 + zernike[(6, 4)] ** 2) * 36
    polar['phi54'] = np.arctan2(zernike[(6, -4)], zernike[(6, 4)]) / 4.
    polar['C56'] = np.sqrt(zernike[(6, -6)] ** 2 + zernike[(6, 6)] ** 2) * 6
    polar['phi56'] = np.arctan2(zernike[(6, -6)], zernike[(6, 6)]) / 6.

    polar['C41'] = np.sqrt(zernike[(5, -1)] ** 2 + zernike[(5, 1)] ** 2) * 50
    polar['phi41'] = np.arctan2(zernike[(5, -1)], zernike[(5, 1)])
    polar['C43'] = np.sqrt(zernike[(5, -3)] ** 2 + zernike[(5, 3)] ** 2) * 25
    polar['phi43'] = np.arctan2(zernike[(5, -3)], zernike[(5, 3)]) / 3.
    polar['C45'] = np.sqrt(zernike[(5, -5)] ** 2 + zernike[(5, 5)] ** 2) * 5
    polar['phi45'] = np.arctan2(zernike[(5, -5)], zernike[(5, 5)]) / 5.

    polar['C30'] = 24 * zernike[(4, 0)] - polar['C50']
    polar['C32'] = np.sqrt((zernike[(4, -2)] - polar['C52'] / 18. * np.sin(2 * polar['phi52'])) ** 2 +
                           (zernike[(4, 2)] - polar['C52'] / 18. * np.cos(2 * polar['phi52'])) ** 2) * 16
    polar['phi32'] = np.arctan2(zernike[(4, -2)] - polar['C52'] / 18. * np.sin(2 * polar['phi52']),
                                zernike[(4, 2)] - polar['C52'] / 18. * np.cos(2 * polar['phi52'])) / 2.
    polar['C34'] = np.sqrt((zernike[(4, -4)] - 5 * polar['C54'] / 36. * np.sin(4 * polar['phi54'])) ** 2 +
                           (zernike[(4, 4)] - 5 * polar['C54'] / 36. * np.cos(4 * polar['phi54'])) ** 2) * 4
    polar['phi34'] = np.arctan2(zernike[(4, -4)] - 5 * polar['C54'] / 36. * np.sin(4 * polar['phi54']),
                                zernike[(4, 4)] - 5 * polar['C54'] / 36. * np.cos(4 * polar['phi54'])) / 4.

    polar['C21'] = np.sqrt((zernike[(3, -1)] - 4 * polar['C41'] / 50. * np.sin(polar['phi41'])) ** 2 +
                           (zernike[(3, 1)] - 4 * polar['C41'] / 50. * np.cos(polar['phi41'])) ** 2) * 9
    polar['phi21'] = np.arctan2(zernike[(3, -1)] - 4 * polar['C41'] / 50. * np.sin(polar['phi41']),
                                zernike[(3, 1)] - 4 * polar['C41'] / 50. * np.cos(polar['phi41']))
    polar['C23'] = np.sqrt((zernike[(3, -3)] - 4 * polar['C43'] / 25. * np.sin(3 * polar['phi43'])) ** 2 +
                           (zernike[(3, 3)] - 4 * polar['C43'] / 25. * np.cos(3 * polar['phi43'])) ** 2) * 3
    polar['phi23'] = np.arctan2(zernike[(3, -3)] - 4 * polar['C43'] / 25. * np.sin(3 * polar['phi43']),
                                zernike[(3, 3)] - 4 * polar['C43'] / 25. * np.cos(3 * polar['phi43'])) / 3.

    polar['C10'] = 4 * zernike[(2, 0)] - polar['C30'] / 2. - 3 / 10. * polar['C50']
    polar['C12'] = np.sqrt((zernike[(2, -2)]
                            - 3 * polar['C32'] / 16. * np.sin(2 * polar['phi32'])
                            - (1 / 6. - 1 / 15.) * polar['C52'] * np.sin(2 * polar['phi52'])) ** 2 +
                           (zernike[(2, 2)]
                            - 3 * polar['C32'] / 16. * np.cos(2 * polar['phi32'])
                            - (1 / 6. - 1 / 15.) * polar['C52'] * np.cos(2 * polar['phi52'])) ** 2) * 2
    polar['phi12'] = np.arctan2(zernike[(2, -2)]
                                - 3 * polar['C32'] / 16. * np.sin(2 * polar['phi32'])
                                - (1 / 6. - 1 / 15.) * polar['C52'] * np.sin(2 * polar['phi52']),
                                zernike[(2, 2)]
                                - 3 * polar['C32'] / 16. * np.cos(2 * polar['phi32'])
                                - (1 / 6. - 1 / 15.) * polar['C52'] * np.cos(2 * polar['phi52'])) / 2.

    for symbol, value in polar.items():
        if symbol[0] == 'C':
            polar[symbol] /= aperture_radius ** (polar_aberration_order(symbol) + 1)

    return {key: value for key, value in polar.items() if value != 0.}


class PhaseAberration(FrequencyTransfer):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True, parametrization='polar',
                 **kwargs):

        if parametrization.lower() == 'polar':
            self._parametrization = PolarAberrations(**kwargs)

        elif parametrization.lower() == 'cartesian':
            self._parametrization = CartesianAberrations(**kwargs)

        else:
            raise RuntimeError()

        FrequencyTransfer.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                   save_tensor=save_tensor)

        self.observe(self._parametrization)

    def set_paramters(self, parameters):
        self._parametrization.set_parameters(parameters)

    @property
    def parametrization(self):
        return self._parametrization

    def line_data(self, phi, k_max=2, n=1024):
        k = tf.linspace(0., k_max, n)
        alpha = self.wavelength * k
        tensor = self.parametrization(alpha=alpha, alpha2=alpha ** 2, phi=phi)

        return k, complex_exponential(2 * np.pi / self.wavelength * tensor)

    def _calculate_tensor(self, alpha=None, alpha2=None, phi=None):
        if isinstance(self._parametrization, PolarAberrations):
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha = tf.sqrt(alpha2)
            phi = angle(alpha_x, alpha_y)

            tensor = self.parametrization(alpha=alpha, alpha2=alpha2, phi=phi)[None, :, :]

        elif isinstance(self._parametrization, CartesianAberrations):
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)
            alpha_x_2 = alpha_x ** 2
            alpha_y_2 = alpha_y ** 2

            tensor = self.parametrization(ax=alpha_x, ay=alpha_y, ax2=alpha_x_2, ay2=alpha_y_2, a2=alpha2)[None, :, :]

        else:
            raise RuntimeError('')

        tensor = complex_exponential(2 * np.pi / self.wavelength * tensor)

        return TensorWithEnergy(tensor=tensor, extent=self.extent, space='fourier', energy=self.energy)


class Translate(FrequencyTransfer):

    def __init__(self, positions=None, extent=None, gpts=None, sampling=None, save_tensor=True):
        FrequencyTransfer.__init__(self, extent=extent, gpts=gpts, sampling=sampling, save_tensor=save_tensor)

        if positions is None:
            positions = (0., 0.)

        self._positions = self._validate_positions(positions)

    def _validate_positions(self, positions):
        if isinstance(positions, (np.ndarray, list, tuple)):
            positions = np.array(positions, dtype=np.float32)
            if positions.shape == (2,):
                positions = positions[None, :]

        return positions

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        old = self.positions
        self._positions = self._validate_positions(positions)
        change = np.any(self._positions != old)
        self.notify_observers({'name': '_positions', 'old': old, 'new': positions, 'change': change})

    def _calculate_tensor(self, kx=None, ky=None):

        if (kx is None) | (ky is None):
            kx, ky = self.fftfreq()
            tensor = complex_exponential(2 * np.pi * (kx[None, :, None] * self.positions[:, 0][:, None, None] +
                                                      ky[None, None, :] * self.positions[:, 1][:, None, None]))

        else:
            tensor = complex_exponential(2 * np.pi * (kx * self.positions[:, 0] + ky * self.positions[:, 1]))

        return Tensor(tf.cast(tensor, tf.complex64), extent=self.extent, space='fourier')


class CTF(FrequencyTransfer):

    def __init__(self, aperture_radius=np.inf, aperture_rolloff=0., focal_spread=0., extent=None, gpts=None,
                 sampling=None, energy=None, save_tensor=True, **kwargs):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)

        self._grid = Grid(extent=extent, sampling=sampling, gpts=gpts)
        self._energy = EnergyProperty(energy=energy)

        self._aberrations = PhaseAberration(extent=extent, save_tensor=save_tensor, **kwargs)
        self._aberrations._grid = self._grid
        self._aberrations._energy = self._energy

        self._aperture = Aperture(radius=aperture_radius, rolloff=aperture_rolloff, save_tensor=save_tensor)
        self._aperture._grid = self._grid
        self._aperture._energy = self._energy

        self._temporal_envelope = TemporalEnvelope(focal_spread=focal_spread, save_tensor=save_tensor)
        self._temporal_envelope._grid = self._grid
        self._temporal_envelope._energy = self._energy

        self.observe(self._aberrations)
        self.observe(self._aberrations.parametrization)
        self.observe(self._aperture)
        self.observe(self._temporal_envelope)
        self.observe(self._energy)
        self.observe(self._grid)

    def line_data(self, phi, k_max=2, n=1024):
        k, tensor = self.aberrations.line_data(phi=phi, k_max=k_max, n=n)
        tensor *= tf.cast(self.temporal_envelope.line_data(k_max=k_max, n=n)[1], tf.complex64)
        return k, tensor

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def aperture(self):
        return self._aperture

    @property
    def temporal_envelope(self):
        return self._temporal_envelope

    def _calculate_tensor(self, alpha=None, alpha2=None, phi=None):
        if (alpha is None) | (alpha2 is None) | (phi is None):
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha = tf.sqrt(alpha2)
            phi = angle(alpha_x, alpha_y)

        tensor = self.aberrations._calculate_tensor(alpha, alpha2, phi).tensorflow()

        tensor *= self.aperture._calculate_tensor(alpha).tensorflow()[0]

        tensor *= self.temporal_envelope._calculate_tensor(alpha).tensorflow()[0]

        return TensorWithEnergy(tensor, extent=self.extent, space='fourier', energy=self.energy)

        # def copy(self):
    #     return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
    #                           energy=self.energy, parametrization=self.parametrization.copy())
