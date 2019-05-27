import numpy as np
import tensorflow as tf

from tensorwaves.utils import complex_exponential, squared_norm, angle
from .aberrations import PolarAberrations, CartesianAberrations
from .bases import FrequencyTransfer, HasGridAndEnergy, HasGrid, TensorWithGridAndEnergy, TensorWithGrid
from .bases import PrismCoefficients, HasEnergy
from .aberrations import PrismPolarAberrations, PrismCartesianAberrations
from .envelope import Aperture, TemporalEnvelope, GaussianEnvelope, PrismAperture, PrismGaussianEnvelope, \
    PrismTemporalEnvelope


class CTFBase(FrequencyTransfer):

    def __init__(self, aberrations, aperture, temporal_envelope, gaussian_envelope, save_tensor=True):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)

        self._aberrations = aberrations
        self._aperture = aperture
        self._temporal_envelope = temporal_envelope
        self._gaussian_envelope = gaussian_envelope

        for depencency in self.get_dependencies():
            self.observe(depencency)

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def aperture(self):
        return self._aperture

    @property
    def temporal_envelope(self):
        return self._temporal_envelope

    @property
    def gaussian_envelope(self):
        return self._gaussian_envelope

    def get_dependencies(self):
        return [self.aberrations, self.aperture, self.temporal_envelope, self.gaussian_envelope]

    def line_profile(self, phi=0., alpha_max=.02, n=1024):
        k, tensor = self.aberrations.line_profile(phi=phi, alpha_max=alpha_max, n=n)

        if self.aperture.cutoff != np.inf:
            tensor *= tf.cast(self.aperture.line_profile(alpha_max=alpha_max, n=n)[1], tf.complex64)

        if self.temporal_envelope.focal_spread > 0.:
            tensor *= tf.cast(self.temporal_envelope.line_profile(alpha_max=alpha_max, n=n)[1], tf.complex64)

        if self.gaussian_envelope.width != np.inf:
            tensor *= tf.cast(self.gaussian_envelope.line_profile(alpha_max=alpha_max, n=n)[1], tf.complex64)

        return k, tensor


class CTF(HasGridAndEnergy, CTFBase):

    def __init__(self, aperture_cutoff=np.inf, aperture_rolloff=0., focal_spread=0., gaussian_envelope_width=np.inf,
                 gaussian_envelope_scale=1., parametrization='polar', extent=None, gpts=None, sampling=None,
                 energy=None, save_tensor=True, grid=None, energy_wrapper=None, **kwargs):

        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, grid=grid,
                                  energy_wrapper=energy_wrapper)


        aperture = Aperture(cutoff=aperture_cutoff, rolloff=aperture_rolloff, save_tensor=save_tensor,
                            grid=self.grid, energy_wrapper=self.energy_wrapper)

        temporal_envelope = TemporalEnvelope(focal_spread=focal_spread, save_tensor=save_tensor, grid=self.grid,
                                             energy_wrapper=self.energy_wrapper)

        gaussian_envelope = GaussianEnvelope(width=gaussian_envelope_width, scale=gaussian_envelope_scale,
                                             save_tensor=save_tensor, grid=self.grid,
                                             energy_wrapper=self.energy_wrapper)

        if parametrization.lower() == 'polar':
            aberrations = PolarAberrations(save_tensor=save_tensor, grid=self.grid,
                                           energy_wrapper=self.energy_wrapper, **kwargs)

        # elif parametrization.lower() == 'cartesian':
        #     self._aberrations = CartesianAberrations(save_tensor=save_tensor, grid=self.grid,
        #                                              energy_wrapper=self.energy_wrapper, **kwargs)

        else:
            raise RuntimeError()

        CTFBase.__init__(self, aberrations=aberrations, aperture=aperture, temporal_envelope=temporal_envelope,
                         gaussian_envelope=gaussian_envelope, save_tensor=save_tensor)

        for depencency in self.get_dependencies():
            self.observe(depencency)

    def get_dependencies(self):
        dependencies = [self.grid, self.energy_wrapper, self.aberrations, self.aperture, self.temporal_envelope,
                        self.gaussian_envelope]
        return dependencies

    def apply(self, waves):
        return waves.apply_frequency_transfer(self)

    def _calculate_tensor(self, *args):

        if not args:
            alpha_x, alpha_y = self.semiangles()
            alpha2 = squared_norm(alpha_x, alpha_y)
            alpha = tf.sqrt(alpha2)
            if isinstance(self.aberrations, PolarAberrations):
                phi = angle(alpha_x, alpha_y)
                args = (alpha, alpha2, phi)

            elif isinstance(self.aberrations, CartesianAberrations):
                alpha_y, alpha_x = tf.meshgrid(alpha_y, alpha_x)
                alpha_x_2 = alpha_x ** 2
                alpha_y_2 = alpha_y ** 2
                args = (alpha_x, alpha_y, alpha_x_2, alpha_y_2, alpha2)

            else:
                raise RuntimeError()

        elif isinstance(self.aberrations, PolarAberrations):
            (alpha, alpha2, phi) = args

        elif isinstance(self.aberrations, CartesianAberrations):
            (alpha_x, alpha_y, alpha_x_2, alpha_y_2, alpha2) = args
            alpha = tf.sqrt(alpha2)

        else:
            raise RuntimeError()

        tensor = self.aberrations.build(*args).tensor()

        if self.aperture.cutoff != np.inf:
            tensor *= tf.cast(self.aperture.build(alpha).tensor(), tf.complex64)

        if self.temporal_envelope.focal_spread > 0.:
            tensor *= tf.cast(self.temporal_envelope.build(alpha2).tensor(), tf.complex64)

        if self.gaussian_envelope.width != np.inf:
            tensor *= tf.cast(self.gaussian_envelope.build(alpha2).tensor(), tf.complex64)

        return TensorWithGridAndEnergy(tensor, extent=self.extent, energy=self.energy, space='fourier')

    # def copy(self):
    #     return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
    #                           energy=self.energy, parametrization=self.parametrization.copy())


class PrismCTF(HasEnergy, PrismCoefficients, CTFBase):

    def __init__(self, aperture_cutoff=np.inf, aperture_rolloff=0., focal_spread=0., gaussian_envelope_width=np.inf,
                 gaussian_envelope_scale=1., parametrization='polar', kx=None, ky=None,
                 energy=None, save_tensor=True, energy_wrapper=None, **kwargs):

        PrismCoefficients.__init__(self, kx=kx, ky=ky, save_tensor=save_tensor)
        HasEnergy.__init__(self, energy=energy, energy_wrapper=energy_wrapper)

        aperture = PrismAperture(cutoff=aperture_cutoff, rolloff=aperture_rolloff, kx=kx, ky=ky,
                                 save_tensor=save_tensor, energy_wrapper=self.energy_wrapper)

        temporal_envelope = PrismTemporalEnvelope(focal_spread=focal_spread, kx=kx, ky=ky,
                                                  save_tensor=save_tensor, energy_wrapper=self.energy_wrapper)

        gaussian_envelope = PrismGaussianEnvelope(width=gaussian_envelope_width, scale=gaussian_envelope_scale,
                                                  kx=kx, ky=ky, save_tensor=save_tensor,
                                                  energy_wrapper=self.energy_wrapper)

        if parametrization.lower() == 'polar':
            aberrations = PrismPolarAberrations(kx=kx, ky=ky, save_tensor=save_tensor,
                                                energy_wrapper=self.energy_wrapper, **kwargs)

            # elif parametrization.lower() == 'cartesian':
            #     self._aberrations = CartesianAberrations(save_tensor=save_tensor, grid=self.grid,
            #                                              energy_wrapper=self.energy_wrapper, **kwargs)

        else:
            raise RuntimeError()

        CTFBase.__init__(self, aberrations=aberrations, aperture=aperture, temporal_envelope=temporal_envelope,
                         gaussian_envelope=gaussian_envelope, save_tensor=save_tensor)

    def _calculate_tensor(self, *args):
        if not args:
            alpha_x, alpha_y = self.kx * self.wavelength, self.ky * self.wavelength
            alpha2 = alpha_x ** 2 + alpha_y ** 2
            alpha = tf.sqrt(alpha2)
            if isinstance(self.aberrations, PrismPolarAberrations):
                phi = tf.atan2(alpha_x, alpha_y)
                args = (alpha, alpha2, phi)

            elif isinstance(self.aberrations, PrismCartesianAberrations):
                alpha_x_2 = alpha_x ** 2
                alpha_y_2 = alpha_y ** 2
                args = (alpha_x, alpha_y, alpha_x_2, alpha_y_2, alpha2)

            else:
                raise RuntimeError()

        elif isinstance(self.aberrations, PrismPolarAberrations):
            (alpha, alpha2, phi) = args

        elif isinstance(self.aberrations, PrismCartesianAberrations):
            (alpha_x, alpha_y, alpha_x_2, alpha_y_2, alpha2) = args
            alpha = tf.sqrt(alpha2)

        else:
            raise RuntimeError()

        tensor = self.aberrations.build(*args)

        if self.aperture.cutoff != np.inf:
            tensor *= tf.cast(self.aperture.build(alpha), tf.complex64)

        if self.temporal_envelope.focal_spread > 0.:
            tensor *= tf.cast(self.temporal_envelope.build(alpha2), tf.complex64)

        if self.gaussian_envelope.width != np.inf:
            tensor *= tf.cast(self.gaussian_envelope.build(alpha2), tf.complex64)

        return tensor

    def copy(self):
        return self.__class__(aperture_cutoff=self.aperture.cutoff, aperture_rolloff=self.aperture.rolloff,
                              focal_spread=self.temporal_envelope.focal_spread,
                              gaussian_envelope_width=self.gaussian_envelope.width,
                              gaussian_envelope_scale=self.gaussian_envelope.scale, kx=self.kx, ky=self.ky,
                              save_tensor=self.save_tensor, energy_wrapper=self.energy_wrapper.copy())


class TranslateBase(FrequencyTransfer):

    def __init__(self, positions=None, save_tensor=True):
        FrequencyTransfer.__init__(self, save_tensor=save_tensor)

        if positions is None:
            positions = (0., 0.)

        self._positions = self._validate_positions(positions)

    def _validate_positions(self, positions):
        if isinstance(positions, (np.ndarray, list, tuple)):
            positions = np.array(positions, dtype=np.float32)
            if positions.shape == (2,):
                positions = positions[None, :]

        else:
            raise RuntimeError('')

        return positions

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        old = self.positions
        self._positions = self._validate_positions(value)
        change = np.any(self._positions != old)
        self.notify_observers({'name': '_positions', 'old': old, 'new': value, 'change': change})

    def _calculate_tensor(self, *args):
        kx, ky = args

        x = tf.reshape(self.positions[:, 0], (-1,) + (1,) * (len(kx.shape) - 1))
        y = tf.reshape(self.positions[:, 1], (-1,) + (1,) * (len(ky.shape) - 1))

        tensor = complex_exponential(2 * np.pi * (kx * x + ky * y))
        return tensor


class Translate(HasGrid, TranslateBase):

    def __init__(self, positions=None, extent=None, gpts=None, sampling=None, save_tensor=True, grid=None):
        TranslateBase.__init__(self, positions=positions, save_tensor=save_tensor)
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid)

    def _calculate_tensor(self, *args):
        if args:
            kx, ky = args

        else:
            kx, ky = self.fftfreq()

        tensor = TranslateBase._calculate_tensor(self, kx[None, :, None], ky[None, None, :])

        return TensorWithGrid(tensor, extent=self.extent, space='fourier')


class PrismTranslate(PrismCoefficients, TranslateBase):

    def __init__(self, position=None, kx=None, ky=None, save_tensor=True):
        TranslateBase.__init__(self, positions=position, save_tensor=save_tensor)
        PrismCoefficients.__init__(self, kx=kx, ky=ky)

    def _calculate_tensor(self, *args):
        tensor = TranslateBase._calculate_tensor(self, self.kx, self.ky)
        return tensor

    def copy(self):
        return self.__class__(position=self.positions, kx=self.kx, ky=self.ky, save_tensor=self.save_tensor)
