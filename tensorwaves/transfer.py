from collections import defaultdict

import numpy as np
import tensorflow as tf

from tensorwaves.bases import TensorWithEnergy, Tensor, notifying_property, named_property, Observable, \
    HasGrid, HasEnergy, TensorFactory, Grid, EnergyProperty
from tensorwaves.bases import complex_exponential
from tensorwaves.ops import squared_norm, angle


class FrequencyTransfer(HasGrid, HasEnergy, TensorFactory, Observable):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        HasEnergy.__init__(self, energy=energy)
        TensorFactory.__init__(self, save_tensor=save_tensor)
        Observable.__init__(self)

        self._grid.register_observer(self)
        self._energy.register_observer(self)
        self.register_observer(self)

    def check_is_defined(self):
        self._grid.check_is_defined()
        self._energy.check_is_defined()

    def match(self, other):
        try:
            self._grid.match(other._grid)
        except AttributeError:
            pass

        try:
            self._energy.match(other._energy)
        except AttributeError:
            pass

    def semiangles(self):
        kx, ky = self._grid.fftfreq()
        wavelength = self.wavelength
        return kx * wavelength, ky * wavelength


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

    def _calculate_tensor(self, alpha=None):
        if alpha is None:
            alpha = tf.sqrt(squared_norm(*self.semiangles()))

        if self.focal_spread > 0.:
            tensor = tf.exp(
                -tf.sign(self.focal_spread) * (
                        .5 * np.pi / self.wavelength * self.focal_spread * alpha ** 2) ** 2)
        else:
            tensor = tf.ones(alpha.shape)

        return TensorWithEnergy(tf.cast(tensor[None, :, :], tf.complex64), extent=self.extent, space='fourier',
                                energy=self.energy)


class PolarAberrations(Observable):

    def __init__(self,
                 C10=0., C12=0., phi12=0.,
                 C21=0., phi21=0., C23=0., phi23=0.,
                 C30=0., C32=0., phi32=0., C34=0., phi34=0.,
                 C41=0., phi41=0., C43=0., phi43=0., C45=0., phi45=0.,
                 C50=0., C52=0., phi52=0., C54=0., phi54=0., C56=0., phi56=0.,
                 defocus=None, astig_mag=None, astig_angle=None, coma=None, coma_angle=None, Cs=None, C5=None):

        Observable.__init__(self)

        if defocus is not None:
            self._C10 = defocus
        else:
            self._C10 = C10
        if astig_mag is not None:
            self._C12 = astig_mag
        else:
            self._C12 = C12
        if astig_angle is not None:
            self._phi12 = astig_angle
        else:
            self._phi12 = phi12

        if coma is not None:
            self._C21 = coma
        else:
            self._C21 = C21
        if coma_angle is not None:
            self._phi21 = coma_angle
        else:
            self._phi21 = phi21
        self._C23 = C23
        self._phi23 = phi23

        if Cs is not None:
            self._C30 = Cs
        else:
            self._C30 = C30
        self._C32 = C32
        self._phi32 = phi32
        self._C34 = C34
        self._phi34 = phi34

        self._C41 = C41
        self._phi41 = phi41
        self._C43 = C43
        self._phi43 = phi43
        self._C45 = C45
        self._phi45 = phi45

        if C5 is not None:
            self._C50 = C5
        else:
            self._C50 = C50
        self._C52 = C52
        self._phi52 = phi52
        self._C54 = C54
        self._phi54 = phi54
        self._C56 = C56
        self._phi56 = phi56

    C10 = notifying_property('_C10')
    C12 = notifying_property('_C12')
    phi12 = notifying_property('_phi12')

    C21 = notifying_property('_C21')
    phi21 = notifying_property('_phi21')
    C23 = notifying_property('_C23')
    phi23 = notifying_property('_phi23')

    C30 = notifying_property('_C30')
    C32 = notifying_property('_C32')
    phi32 = notifying_property('_phi32')
    C34 = notifying_property('_C34')
    phi34 = notifying_property('_phi34')

    C41 = notifying_property('_C41')
    phi41 = notifying_property('_phi41')
    C43 = notifying_property('_C43')
    phi43 = notifying_property('_phi43')
    C45 = notifying_property('_C45')
    phi45 = notifying_property('_phi45')

    C50 = notifying_property('_C50')
    C52 = notifying_property('_C52')
    phi52 = notifying_property('_phi52')
    C54 = notifying_property('_C54')
    phi54 = notifying_property('_phi54')
    C56 = notifying_property('_C56')
    phi56 = notifying_property('_phi56')

    defocus = named_property('C10')
    astig_mag = named_property('C12')
    astig_angle = named_property('phi12')
    coma = named_property('C21')
    coma_angle = named_property('phi21')
    Cs = named_property('C30')
    C5 = named_property('C50')

    def set_parameters(self, parameters):

        for name, value in parameters.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise RuntimeError()

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


class CartesianAberrations(Observable):

    def __init__(self,
                 C10=0., C12a=0., C12b=0.,
                 C21a=0., C21b=0., C23a=0., C23b=0.,
                 C30=0., C32a=0., C32b=0., C34a=0., C34b=0.,
                 defocus=None, astig_x=None, astig_y=None, coma_x=None, coma_y=None, Cs=None):
        Observable.__init__(self)

        if defocus is not None:
            self._C10 = defocus
        else:
            self._C10 = C10
        if astig_x is not None:
            self._C12a = astig_x
        else:
            self._C12a = C12a
        if astig_y is not None:
            self._C12b = astig_y
        else:
            self._C12b = C12b

        if coma_x is not None:
            self._C21a = coma_x
        else:
            self._C21a = C21a
        if coma_y is not None:
            self._C21b = coma_y
        else:
            self._C21b = C21b
        self._C23a = C23a
        self._C23b = C23b

        if Cs is not None:
            self._C30 = Cs
        else:
            self._C30 = C30
        self._C32a = C32a
        self._C32b = C32b
        self._C34a = C34a
        self._C34b = C34b

    C10 = notifying_property('_C10')
    C12a = notifying_property('_C12a')
    C12b = notifying_property('_C12b')

    C21a = notifying_property('_C21a')
    C21b = notifying_property('_C21b')
    C23a = notifying_property('_C23a')
    C23b = notifying_property('_C23b')

    C30 = notifying_property('_C30')
    C32a = notifying_property('_C32a')
    C32b = notifying_property('_C32b')
    C34a = notifying_property('_C34a')
    C34b = notifying_property('_C34b')

    defocus = named_property('C10')
    astig_x = named_property('C12a')
    astig_y = named_property('C12b')
    coma_x = named_property('C21a')
    coma_y = named_property('C21b')
    Cs = named_property('C30')
    C5 = named_property('C50')

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

    return {key: value for key, value in cartesian.items() if value != 0.}


def cartesian2polar(cartesian):
    cartesian = defaultdict(lambda: 0, cartesian)

    polar = {}
    polar['C10'] = cartesian['C10']
    polar['C12'] = - cartesian['C12a'] * np.sqrt(1 + (cartesian['C12b'] / cartesian['C12a']) ** 2)
    polar['phi12'] = - np.arctan(cartesian['C12b'] / cartesian['C12a']) / 2.
    polar['C21'] = cartesian['C21b'] * np.sqrt(1 + (cartesian['C21a'] / cartesian['C21b']) ** 2)
    polar['phi21'] = np.arctan(cartesian['C21a'] / cartesian['C21b'])
    polar['C23'] = cartesian['C23b'] * np.sqrt(1 + (cartesian['C23a'] / cartesian['C23b']) ** 2)
    polar['phi23'] = -np.arctan(cartesian['C23a'] / cartesian['C23b']) / 3.
    polar['C30'] = cartesian['C30']
    polar['C32'] = -cartesian['C32a'] * np.sqrt(1 + (cartesian['C32b'] / cartesian['C32a']) ** 2)
    polar['phi32'] = -np.arctan(cartesian['C32b'] / cartesian['C32a']) / 2.
    K = np.sqrt(3 + np.sqrt(8.))
    A = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K)
    B = 4 * np.arctan(1 / K)
    polar['phi34'] = np.arctan(1 / (A * np.sin(B)) * cartesian['C34b'] / cartesian['C34a'] - 1 / np.tan(B)) / 4
    polar['C34'] = cartesian['C34a'] / np.cos(-4 * polar['phi34'])

    return {key: value for key, value in polar.items() if value != 0.}


class PhaseAberration(FrequencyTransfer):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True, parametrization='polar',
                 **kwargs):

        if parametrization.lower() == 'polar':
            self._parameters = PolarAberrations(**kwargs)

        elif parametrization.lower() == 'cartesian':
            self._parameters = CartesianAberrations(**kwargs)

        else:
            raise RuntimeError()

        FrequencyTransfer.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                   save_tensor=save_tensor)

    @property
    def parameters(self):
        return self._parameters

    def _line_data(self, phi, k_max=2, n=1024):
        k = tf.linspace(0., k_max, n)
        alpha = self.wavelength * k
        tensor = self.parameters(alpha=alpha, alpha2=alpha ** 2, phi=phi)

        return k, complex_exponential(2 * np.pi / self.wavelength * tensor)

    def _calculate_tensor(self, alpha=None, alpha2=None, phi=None):

        if isinstance(self._parameters, PolarAberrations):
            # if (alpha is None) | (alpha2 is None) | (phi is None):
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha = tf.sqrt(alpha2)
            phi = angle(alpha_x, alpha_y)

            tensor = self.parameters(alpha=alpha, alpha2=alpha2, phi=phi)[None, :, :]

        elif isinstance(self._parameters, CartesianAberrations):
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)
            alpha_x_2 = alpha_x ** 2
            alpha_y_2 = alpha_y ** 2

            tensor = self.parameters(ax=alpha_x, ay=alpha_y, ax2=alpha_x_2, ay2=alpha_y_2, a2=alpha2)[None, :, :]

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

        self._aberrations.register_observer(self)
        self._aperture.register_observer(self)
        self._temporal_envelope.register_observer(self)
        self._energy.register_observer(self)
        self._grid.register_observer(self)

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

        # print(self.aberrations.defocus)

        tensor = self.aberrations._calculate_tensor(alpha, alpha2, phi).tensorflow()

        tensor *= self.aperture._calculate_tensor(alpha).tensorflow()[0]

        tensor *= self.temporal_envelope._calculate_tensor(alpha).tensorflow()[0]

        return TensorWithEnergy(tensor, extent=self.extent, space='fourier', energy=self.energy)

        # def copy(self):
    #     return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
    #                           energy=self.energy, parametrization=self.parametrization.copy())
