from math import pi

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorwaves import utils
from tensorwaves.bases import FactoryBase, TensorBase


class CTF(FactoryBase):

    def __init__(self, parametrization='polar', parameters=None, energy=None, gpts=None, extent=None, sampling=None,
                 **args):
        super().__init__(energy=energy, gpts=gpts, extent=extent, sampling=sampling)

        if parametrization is 'polar':
            symbols = polar_symbols
            aliases = polar_aliases

        elif parametrization is 'cartesian':
            symbols = cartesian_symbols
            aliases = cartesian_aliases

        else:
            raise ValueError()

        self._parametrization = parametrization

        if parameters is None:
            parameters = {}

        self._parameters = dict(zip(symbols, [0.] * len(symbols)))
        self._parameters.update(parameters)
        self._parameters.update(**args)

        for alias, symbol in zip(aliases, symbols):
            if (alias is not None):
                if alias not in self._parameters.keys():
                    self._parameters[alias] = self._parameters[symbol]
                else:
                    self._parameters[symbol] = self._parameters[alias]

        self._func = polar_ctf(self._parameters)

    @property
    def parameters(self):
        return self._parameters

    def __repr__(self):
        return 'contrast transfer function\n' + super().__repr__()

    def show_radial(self, phi=0, k_max=2, n=1024, ax=None):
        if ax is None:
            ax = plt.subplot()

        wavelength = utils.energy2wavelength(self._energy)

        k = np.linspace(0, k_max, n)
        alpha = wavelength * k
        chi = 2 * pi / wavelength * self._func(alpha, alpha ** 2, phi)

        ax.plot(k, np.sin(chi))
        ax.set_xlabel('k [1 / Angstrom]')

        return ax

    def _tensor(self):
        wavelength = utils.energy2wavelength(self.energy)
        kx, ky = self.fftfreq()

        alpha_x = kx * wavelength
        alpha_y = ky * wavelength

        alpha2 = alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2
        alpha = tf.sqrt(alpha2)
        alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)

        if self._parametrization == 'polar':
            phi = tf.atan2(alpha_x, alpha_y)
            chi = self._func(alpha, alpha2, phi)
        else:
            raise NotImplementedError()

        return utils.complex_exponential(2 * pi / wavelength * chi)

    def build(self):
        return TensorBase(tensor=self._tensor(), energy=self.energy, extent=self.extent, sampling=self.sampling)


polar_symbols = ('C10', 'C12', 'phi12',
                 'C21', 'phi21', 'C23', 'phi23',
                 'C30', 'C32', 'phi32', 'C34', 'phi34',
                 'C41', 'phi41', 'C43', 'phi43', 'C45', 'phi45',
                 'C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')

# todo: alias for higher order aberrations
polar_aliases = ('defocus', 'astig_mag', 'astig_angle',
                 'coma_mag', 'coma_angle', 'astig_mag_2', 'astig_angle_2',
                 'Cs', None, None, None, None,
                 None, None, None, None, None, None,
                 'C5', None, None, None, None, None, None)


def symmetric_ctf(p):
    if p['C10'] != 0.:
        chi = lambda a, a2: 1 / 2. * a2 * p['C10']
    if p['C30'] != 0.:
        chi_old1 = chi  # todo: better way to avoid recursion?
        chi = lambda a, a2: chi_old1(a, a2) + 1 / 4. * a2 ** 2 * p['C30']
    if p['C50'] != 0.:
        chi_old2 = chi
        chi = lambda a, a2: chi_old2(a, a2) + 1 / 6. * a2 ** 3 * p['C50']
    return chi


def polar_ctf(p):
    # todo: this can be more optimized, although the gains will be small
    if any([p[key] != 0. for key in ('C10', 'C12', 'phi12')]):
        chi = lambda a, a2, b: (1 / 2. * a2 *
                                (p['C10'] +
                                 p['C12'] * tf.cos(2. * (b - p['phi12']))))
    if any([p[key] != 0. for key in ('C21', 'phi21', 'C23', 'phi23')]):
        chi_old1 = chi
        chi = lambda a, a2, b: (chi_old1(a, a2, b) + 1 / 3. * a2 * a *
                                (p['C21'] * tf.cos(b - p['phi21']) +
                                 p['C23'] * tf.cos(3. * (b - p['phi23']))))
    if any([p[key] != 0. for key in ('C30', 'C32', 'phi32', 'C34', 'phi34')]):
        chi_old2 = chi
        chi = lambda a, a2, b: (chi_old2(a, a2, b) + 1 / 4. * a2 ** 2 *
                                (p['C30'] +
                                 p['C32'] * tf.cos(2. * (b - p['phi32'])) +
                                 p['C34'] * tf.cos(4. * (b - p['phi34']))))
    if any([p[key] != 0. for key in ('C41', 'phi41', 'C43', 'phi43', 'C45', 'phi41')]):
        chi_old3 = chi
        chi = lambda a, a2, b: (chi_old3(a, a2, b) + 1 / 5. * a2 ** 2 * a *
                                (p['C41'] * tf.cos((b - p['phi41'])) +
                                 p['C43'] * tf.cos(3. * (b - p['phi43'])) +
                                 p['C45'] * tf.cos(5. * (b - p['phi41']))))
    if any([p[key] != 0. for key in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
        chi_old4 = chi
        chi = lambda a, a2, b: (chi_old4(a, a2, b) + 1 / 6. * a2 ** 3 *
                                (p['C50'] +
                                 p['C52'] * tf.cos(2. * (b - p['phi52'])) +
                                 p['C54'] * tf.cos(4. * (b - p['phi54'])) +
                                 p['C56'] * tf.cos(6. * (b - p['phi56']))))
    try:
        return chi
    except UnboundLocalError:
        return lambda a, a2, b: 0


cartesian_symbols = ('C10', 'C12a', 'C12b',
                     'C21a', 'C21b', 'C23a', 'C23b',
                     'C30', 'C32a', 'C32b', 'C34a', 'C34b')

cartesian_aliases = ('defocus', 'astig_x', 'astig_y',
                     'coma_x', 'coma_y', 'astig_x_2', 'astig_y_2',
                     'Cs', None, None, None, None)


def cartesian_ctf(p):
    # todo: implement 4th and 5th order
    if any([p[key] != 0. for key in ('C10', 'C12a', 'C12b')]):
        chi = lambda ax, ay, ax2, ay2, a2: (1 / 2. * (p['C10'] * a2 +
                                                      p['C12a'] * (ax2 - ay2)) + p['C12b'] * ax * ay)
    if any([p[key] != 0. for key in ('C21a', 'C21b', 'C23a', 'C23b')]):
        chi_old1 = chi
        chi = (lambda ax, ay, ax2, ay2, a2: chi_old1(ax, ay, ax2, ay2, a2) +
                                            1 / 3. * (a2 * (p['C21a'] * ax + p['C21b'] * ay) +
                                                      p['C23a'] * ax * (ax2 - 3 * ay2) +
                                                      p['C23b'] * ay * (ay2 - 3 * ax2)))
    if any([p[key] != 0. for key in ('C30', 'C32a', 'C32b', 'C34a', 'C34b')]):
        chi_old2 = chi
        chi = (lambda ax, ay, ax2, ay2, a2: chi_old2(ax, ay, ax2, ay2, a2) +
                                            1 / 4. * (p['C30'] * a2 ** 2 +
                                                      p['C32a'] * (ax2 ** 2 - ay2 ** 2) +
                                                      2 * p['C32b'] * ax * ay * a2 +
                                                      p['C34a'] * (ax2 ** 2 - 6 * ax2 * ay2 + ay2 ** 2) +
                                                      4 * p['C34b'] * (ax * ay2 * ay - ax2 * ax * ay)))
    return chi


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


def cartesian2polar(polar):
    # todo: implement this function
    raise NotImplementedError()
