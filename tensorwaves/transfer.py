from math import pi

import numpy as np
import tensorflow as tf

from tensorwaves.bases import FrequencyMultiplier, TensorWithEnergy, Tensor, notifying_property, base_property, \
    HasData, HasAccelerator
from tensorwaves.utils import complex_exponential


class Aperture(FrequencyMultiplier):

    def __init__(self, radius=np.inf, rolloff=0., extent=None, gpts=None, sampling=None, energy=None,
                 save_data=True, grid=None, accelerator=None):
        FrequencyMultiplier.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                     save_data=save_data, grid=grid, accelerator=accelerator)

        self._radius = radius
        self._rolloff = rolloff

    radius = notifying_property('_radius')
    rolloff = notifying_property('_rolloff')

    def _calculate_data(self, alpha=None):
        if alpha is None:
            _, _, alpha2 = self.get_semiangles(return_squared_norm=True)
            alpha = tf.sqrt(alpha2)

        if self.rolloff > 0.:
            tensor = .5 * (1 + tf.cos(pi * (alpha - self.radius) / self.rolloff))
            tensor *= tf.cast(alpha < self.radius + self.rolloff, tf.float32)
            tensor = tf.where(alpha > self.radius, tensor, tf.ones(alpha.shape))
        else:
            tensor = tf.cast(alpha < self.radius, tf.float32)

        return TensorWithEnergy(tf.cast(tensor[None, :, :], tf.complex64), extent=self.grid.extent, space=self.space,
                                energy=self.accelerator.energy)

    def copy(self):
        return self.__class__(radius=self.radius, rolloff=self.rolloff, save_data=True, grid=self.grid.copy(),
                              accelerator=self.accelerator.copy())


class PrismAperture(HasData, HasAccelerator):

    def __init__(self, kx, ky, radius=np.inf, rolloff=0., energy=None, accelerator=None, save_data=True, **kwargs):
        self._kx = kx
        self._ky = ky

        self._radius = radius
        self._rolloff = rolloff

        HasData.__init__(self, save_data=save_data)
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        self.register_observer(self)

    radius = notifying_property('_radius')
    rolloff = notifying_property('_rolloff')

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

    def _calculate_data(self):
        alpha_x = self.kx * self.accelerator.wavelength
        alpha_y = self.ky * self.accelerator.wavelength

        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)

        if self.rolloff > 0.:
            tensor = .5 * (1 + tf.cos(pi * (alpha - self.radius) / self.rolloff))
            tensor *= tf.cast(alpha < self.radius + self.rolloff, tf.float32)
            tensor = tf.where(alpha > self.radius, tensor, tf.ones(alpha.shape))
        else:
            tensor = tf.cast(alpha < self.radius, tf.float32)
        # tensor = tf.cast(alpha < self.radius, tf.float32)
        return tensor

        #
        # phi = tf.atan2(alpha_x, alpha_y)
        #
        # tensor = PolarAbberations._calculate_data(self, alpha=alpha, alpha2=alpha2, phi=phi)
        #
        # tensor = complex_exponential(
        #     - 2 * np.pi / self.accelerator.wavelength * tensor)
        #
        # return tensor


class TemporalEnvelope(FrequencyMultiplier):

    def __init__(self, focal_spread=0., extent=None, gpts=None, sampling=None, energy=None,
                 save_data=True, grid=None, accelerator=None):

        FrequencyMultiplier.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                     save_data=save_data, grid=grid, accelerator=accelerator)

        self._focal_spread = np.float32(focal_spread)

    focal_spread = notifying_property('_focal_spread')

    def _calculate_data(self, alpha=None):
        if alpha is None:
            _, _, alpha2 = self.get_semiangles(return_squared_norm=True)
            alpha = tf.sqrt(alpha2)

        if self.focal_spread > 0.:
            tensor = tf.exp(
                -tf.sign(self.focal_spread) * (
                        .5 * np.pi / self.accelerator.wavelength * self.focal_spread * alpha ** 2) ** 2)
        else:
            tensor = tf.ones(alpha.shape)

        return TensorWithEnergy(tf.cast(tensor[None, :, :], tf.complex64), extent=self.grid.extent, space=self.space,
                                energy=self.accelerator.energy)


class PolarAberrations(object):

    def __init__(self,
                 C10=0., C12=0., phi12=0.,
                 C21=0., phi21=0., C23=0., phi23=0.,
                 C30=0., C32=0., phi32=0., C34=0., phi34=0.,
                 C41=0., phi41=0., C43=0., phi43=0., C45=0., phi45=0.,
                 C50=0., C52=0., phi52=0., C54=0., phi54=0., C56=0., phi56=0.,
                 defocus=0., astig_mag=0., astig_angle=0., coma=0., coma_angle=0., Cs=0., C5=0.):

        self._C10 = C10
        self._C12 = C12
        self._phi12 = phi12

        self._C21 = C21
        self._phi21 = phi21
        self._C23 = C23
        self._phi23 = phi23

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

        self._C50 = C50
        self._C52 = C52
        self._phi52 = phi52
        self._C54 = C54
        self._phi54 = phi54
        self._C56 = C56
        self._phi56 = phi56

        self._C10 = defocus
        self._C12 = astig_mag
        self._phi12 = astig_angle
        self._C21 = coma
        self._phi21 = coma_angle
        self._C30 = Cs
        self._C50 = C5

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

    defocus = base_property('C10')
    astig_mag = base_property('C12')
    astig_angle = base_property('phi12')
    coma = base_property('C21')
    coma_angle = base_property('phi21')
    astig_mag_2 = base_property('C23')
    astig_angle_2 = base_property('phi23')
    Cs = base_property('C30')
    C5 = base_property('C50')

    def _calculate_data(self, alpha, alpha2, phi):

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
                        self.C45 * tf.cos(5. * (phi - self.phi41))))

        if any([getattr(self, symbol) != 0. for symbol in ('C50', 'C52', 'phi52', 'C54', 'phi54', 'C56', 'phi56')]):
            tensor += (1 / 6. * alpha2 ** 3 *
                       (self.C50 +
                        self.C52 * tf.cos(2. * (phi - self.phi52)) +
                        self.C54 * tf.cos(4. * (phi - self.phi54)) +
                        self.C56 * tf.cos(6. * (phi - self.phi56))))

        return tensor


class PhaseAberration(PolarAberrations, FrequencyMultiplier):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_data=True, grid=None, accelerator=None,
                 **kwargs):
        PolarAberrations.__init__(self, **kwargs)

        FrequencyMultiplier.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                     save_data=save_data, grid=grid, accelerator=accelerator)

    def _line_data(self, phi, k_max=2):
        k = tf.linspace(0., k_max, 1024)
        alpha = self.accelerator.wavelength * k
        tensor = PolarAberrations._calculate_data(self, alpha=alpha, alpha2=alpha ** 2, phi=phi)
        return k, complex_exponential(2 * pi / self.accelerator.wavelength * tensor)

    def _calculate_data(self, alpha=None, alpha2=None, phi=None):
        if alpha is None:
            _, _, alpha2, phi = self.get_semiangles(return_squared_norm=True, return_azimuth=True)
            alpha = tf.sqrt(alpha2)

        tensor = PolarAberrations._calculate_data(self, alpha=alpha, alpha2=alpha2, phi=phi)[None, :, :]
        tensor = complex_exponential(2 * pi / self.accelerator.wavelength * tensor)
        return TensorWithEnergy(tensor=tensor, extent=self.grid.extent, space=self.space,
                                energy=self.accelerator.energy)


class PrismAberration(HasData, HasAccelerator, PolarAberrations):

    def __init__(self, kx, ky, energy=None, accelerator=None, save_data=True, **kwargs):
        self._kx = kx
        self._ky = ky

        HasData.__init__(self, save_data=save_data)
        PolarAberrations.__init__(self, **kwargs)
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        self.register_observer(self)

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

    def _calculate_data(self):
        alpha_x = self.kx * self.accelerator.wavelength
        alpha_y = self.ky * self.accelerator.wavelength

        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)

        phi = tf.atan2(alpha_x, alpha_y)

        tensor = PolarAberrations._calculate_data(self, alpha=alpha, alpha2=alpha2, phi=phi)

        tensor = complex_exponential(
            - 2 * np.pi / self.accelerator.wavelength * tensor)

        return tensor


class Translate(FrequencyMultiplier):

    def __init__(self, positions=None, extent=None, gpts=None, sampling=None, save_data=True, grid=None,
                 accelerator=None):
        FrequencyMultiplier.__init__(self, extent=extent, gpts=gpts, sampling=sampling, save_data=save_data,
                                     grid=grid, accelerator=accelerator)

        if positions is None:
            positions = [0., 0.]

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

    def _calculate_data(self, kx=None, ky=None):
        if (kx is None) | (ky is None):
            kx, ky = self.grid.fftfreq()
            tensor = complex_exponential(2 * np.pi * (kx[None, :, None] * self.positions[:, 0][:, None, None] +
                                                      ky[None, None, :] * self.positions[:, 1][:, None, None]))

        else:
            tensor = complex_exponential(2 * np.pi * (kx * self.positions[:, 0] + ky * self.positions[:, 1]))

        return Tensor(tf.cast(tensor, tf.complex64), extent=self.grid.extent, space=self.space)


class PrismTranslate(HasData, HasAccelerator):

    def __init__(self, kx, ky, positions=None, save_data=True, energy=None, accelerator=None):

        self._kx = kx
        self._ky = ky

        HasData.__init__(self, save_data=save_data)
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        if positions is None:
            positions = [0., 0.]

        self._positions = self._validate_positions(positions)

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

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

        return positions

    def _calculate_data(self):
        tensor = complex_exponential(2 * np.pi * (self.kx * self.positions[:, 0] + self.ky * self.positions[:, 1]))

        return tf.cast(tensor, tf.complex64)


class CTF(FrequencyMultiplier):

    def __init__(self, aperture_radius=np.inf, aperture_rolloff=0., focal_spread=0.,
                 extent=None, gpts=None, sampling=None, energy=None, save_data=True, grid=None, accelerator=None,
                 **kwargs):
        self._aberrations = PhaseAberration(extent=extent, gpts=gpts, sampling=sampling, energy=energy,
                                            save_data=save_data, grid=grid, accelerator=accelerator, **kwargs)

        self._aperture = Aperture(radius=aperture_radius, rolloff=aperture_rolloff, save_data=save_data,
                                  grid=self._aberrations.grid, accelerator=self._aberrations.accelerator)

        self._temporal_envelope = TemporalEnvelope(focal_spread=focal_spread, save_data=save_data,
                                                   grid=self._aberrations.grid,
                                                   accelerator=self._aberrations.accelerator)

        FrequencyMultiplier.__init__(self, save_data=save_data, grid=self._aberrations.grid,
                                     accelerator=self._aberrations.accelerator)

        # self._aberrations.register_observer(self)
        # self._aperture.register_observer(self)
        # self._temporal_envelope.register_observer(self)

        self._observing = [self._aberrations, self._aperture, self._temporal_envelope]

    def register_observer(self, observer):
        self._observers.append(observer)
        if not observer in self._aberrations._observers:
            self._aberrations.register_observer(observer)

        if not observer in self._aperture._observers:
            self._aperture.register_observer(observer)

        if not observer in self._temporal_envelope._observers:
            self._temporal_envelope.register_observer(observer)

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def aperture(self):
        return self._aperture

    @property
    def temporal_envelope(self):
        return self._temporal_envelope

    def _calculate_data(self, alpha=None, alpha2=None, phi=None):
        if alpha is None:
            _, _, alpha2, phi = self.get_semiangles(return_squared_norm=True, return_azimuth=True)
            alpha = tf.sqrt(alpha2)

        # tensor = complex_exponential(2 * pi / self.accelerator.wavelength * self._aberrations.get_data()._tensor)
        tensor = self._aberrations.get_data()._tensor

        tensor *= self._aperture.get_data()._tensor[0]

        tensor *= self._temporal_envelope.get_data()._tensor[0]

        return TensorWithEnergy(tensor, extent=self.grid.extent, space=self.space,
                                energy=self.accelerator.energy)

        # def copy(self):
    #     return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
    #                           energy=self.energy, parametrization=self.parametrization.copy())
