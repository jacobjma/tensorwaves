from collections import OrderedDict

import numpy as np
import tensorflow as tf
from ase import Atoms

from tensorwaves.bases import HasGrid, HasEnergy, HasGridAndEnergy, TensorFactory, TensorWithEnergy, notifying_property, \
    Grid, EnergyProperty
from tensorwaves.potentials import Potential
from tensorwaves.prism import PrismAperture, PrismAberration, PrismTranslate
from tensorwaves.scan import LineScan, GridScan
from tensorwaves.transfer import CTF, Translate
from tensorwaves.utils import ProgressBar, complex_exponential, fourier_propagator


class TensorWaves(TensorWithEnergy):

    def __init__(self, tensor, extent=None, sampling=None, energy=None):
        TensorWithEnergy.__init__(self, tensor, extent=extent, sampling=sampling, energy=energy, space='direct')

    def multislice(self, potential, in_place=False, progress_tracker=None):

        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match(potential)

        self.check_is_defined()
        potential.check_is_defined()

        if in_place:
            wave = self
        else:
            wave = self.copy()

        progress_bar = ProgressBar(num_iter=potential.num_slices, description='Multislice')
        #
        # if progress_tracker is not None:
        #     progress_tracker.add_bar(progress_bar)

        for i, potential_slice in enumerate(potential.slice_generator()):
            progress_bar.update(i)

            wave._tensorflow = wave._tensorflow * complex_exponential(wave.sigma * potential_slice)
            wave.propagate(potential.slice_thickness)
        # if progress_tracker is not None:
        #     del progress_tracker._output[progress_bar]

        return wave

    def propagate(self, dz):
        self._tensorflow = self._fourier_convolution(self.fourier_propagator(dz))

    def fourier_propagator(self, dz):
        kx, ky = self.fftfreq()
        return fourier_propagator(((kx ** 2)[:, None] + (ky ** 2)[None, :]), dz, self.wavelength)[None, :, :]

    def _fourier_convolution(self, propagator):
        return tf.signal.ifft2d(tf.signal.fft2d(self._tensorflow) * propagator)

    def apply_frequency_transfer(self, frequency_transfer, in_place=False):
        self.match(frequency_transfer)

        self.check_is_defined()
        frequency_transfer.check_is_defined()

        tensor = tf.signal.ifft2d(tf.signal.fft2d(self._tensorflow) * frequency_transfer.get_tensor().tensorflow())

        if in_place:
            self._tensorflow = tensor

            return self
        else:
            return TensorWaves(tensor, extent=self.extent.copy(), energy=self.energy)

    def apply_ctf(self, ctf=None, in_place=False, aperture_radius=np.inf, aperture_rolloff=0., **kwargs):
        if ctf is None:
            ctf = CTF(aperture_radius=aperture_radius, aperture_rolloff=aperture_rolloff, **kwargs)

        return self.apply_frequency_transfer(frequency_transfer=ctf, in_place=in_place)

    def intensity(self):
        return tf.abs(self._tensorflow) ** 2

    def image(self):
        from tensorwaves.image import Image
        return Image(self.intensity(), extent=self.extent.copy())

    def copy(self):
        return self.__class__(tensor=tf.identity(self._tensorflow), extent=self.extent.copy(), energy=self.energy)


class WaveFactory(HasGridAndEnergy, HasGrid, HasEnergy, TensorFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        HasGridAndEnergy.__init__(self)
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        HasEnergy.__init__(self, energy=energy)
        TensorFactory.__init__(self, save_tensor=save_tensor)

        self._grid.register_observer(self)
        self._energy.register_observer(self)

    def multislice(self, potential, in_place=False):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match(potential)

        return self.get_tensor().multislice(potential, in_place=in_place)


class PlaneWaves(WaveFactory):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_tensor=save_tensor)

        self._num_waves = num_waves

    num_waves = notifying_property('_num_waves')

    def _calculate_tensor(self):
        return TensorWaves(tf.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=tf.complex64),
                           extent=self.extent, energy=self.energy)


class ProbeWaves(WaveFactory):

    def __init__(self, positions=None, aperture_radius=1, aperture_rolloff=0., extent=None, gpts=None, sampling=None,
                 energy=None, save_tensor=True, **kwargs):
        WaveFactory.__init__(self, save_tensor=save_tensor)

        self._grid = Grid(extent=extent, sampling=sampling, gpts=gpts)
        self._energy = EnergyProperty(energy=energy)

        self._ctf = CTF(aperture_radius=aperture_radius, aperture_rolloff=aperture_rolloff, extent=extent, gpts=gpts,
                        sampling=sampling, energy=energy, **kwargs)
        self.ctf._grid = self._grid
        self.ctf._energy = self._energy
        self.aberrations._grid = self._grid
        self.aberrations._energy = self._energy
        self.aperture._grid = self._grid
        self.aperture._energy = self._energy
        self.temporal_envelope._grid = self._grid
        self.temporal_envelope._energy = self._energy

        self._translate = Translate(positions=positions)
        self.translate._grid = self._grid
        self.translate._energy = self._energy

        self.ctf.register_observer(self)
        self.ctf.aberrations.register_observer(self)
        self.ctf.aperture.register_observer(self)
        self.ctf.temporal_envelope.register_observer(self)
        self.translate.register_observer(self)
        self._grid.register_observer(self)
        self._energy.register_observer(self)

    @property
    def ctf(self):
        return self._ctf

    @property
    def aberrations(self):
        return self.ctf._aberrations

    @property
    def aperture(self):
        return self._ctf._aperture

    @property
    def temporal_envelope(self):
        return self._ctf._temporal_envelope

    @property
    def translate(self):
        return self._translate

    @property
    def positions(self):
        return self._translate.positions

    @positions.setter
    def positions(self, value):
        self._translate.positions = value

    def linescan(self, start, end, num_positions=None, sampling=None, endpoint=True, detectors=None, potential=None,
                 max_batch=1):
        scan = LineScan(scanable=self, detectors=detectors, start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        scan.scan(max_batch=max_batch, potential=potential)
        return scan

    def gridscan(self, start=None, end=None, num_positions=None, sampling=None, endpoint=False, max_batch=1,
                 potential=None, detectors=None):
        scan = GridScan(scanable=self, detectors=detectors, start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        scan.scan(max_batch=max_batch, potential=potential)

        return scan

    def _calculate_tensor(self):
        return TensorWaves(
            tf.signal.fft2d(self.ctf.get_tensor().tensorflow() * self.translate.get_tensor().tensorflow()),
            extent=self.extent, energy=self.energy)


class PrismWaves(WaveFactory):

    def __init__(self, cutoff=0.01, interpolation=1., gpts=None, extent=None, sampling=None, energy=None,
                 save_tensor=True):
        self.cutoff = cutoff
        self.interpolation = interpolation

        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_tensor=save_tensor)

    def get_scattering_matrix(self):
        return self.get_tensor()

    def _calculate_tensor(self):
        n_max = np.ceil(self.cutoff / (self.wavelength / self.extent[0] * self.interpolation))
        m_max = np.ceil(self.cutoff / (self.wavelength / self.extent[1] * self.interpolation))

        kx = tf.cast(tf.range(-n_max, n_max + 1), tf.float32) / self.extent[0] * self.interpolation
        ky = tf.cast(tf.range(-m_max, m_max + 1), tf.float32) / self.extent[1] * self.interpolation

        mask = tf.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) < (self.cutoff / self.wavelength)

        ky, kx = tf.meshgrid(ky, kx)

        kx = tf.boolean_mask(kx, mask)
        ky = tf.boolean_mask(ky, mask)

        x, y = self.linspace()

        tensor = complex_exponential(2 * np.pi * (kx[:, None, None] * x[None, :, None] +
                                                  ky[:, None, None] * y[None, None, :]))

        return ScatteringMatrix(tensor, kx=kx, ky=ky, interpolation=self.interpolation, extent=self.extent,
                                energy=self.energy)


def wrapped_slice(tensor, begin, size):
    shift = [-x for x in begin]
    tensor = tf.roll(tensor, shift, list(range(len(begin))))
    return tf.slice(tensor, [0] * len(begin), size)


class ScatteringMatrix(TensorWaves, TensorFactory):

    def __init__(self, expansion, kx, ky, interpolation, position=None, extent=None, sampling=None, energy=None,
                 save_tensor=True, **kwargs):

        if kx.shape != ky.shape:
            raise RuntimeError('')

        if expansion.shape[0] != kx.shape[0]:
            raise RuntimeError('')

        self._kx = kx
        self._ky = ky
        self._interpolation = interpolation

        TensorWaves.__init__(self, tensor=expansion, extent=extent, sampling=sampling)
        TensorFactory.__init__(self, save_tensor=save_tensor)

        self._energy = EnergyProperty(energy=energy)

        self._aberrations = PrismAberration(kx=kx, ky=ky, **kwargs)
        self.aberrations._energy = self._energy

        self._translate = PrismTranslate(kx=kx, ky=ky, position=position)
        self._translate._energy = self._energy

        self._aperture = PrismAperture(kx=kx, ky=ky, radius=np.inf)
        self._aperture._energy = self._energy

        self._translate.register_observer(self)
        self._aberrations.register_observer(self)
        self._aberrations._parametrization.register_observer(self)
        self._aperture.register_observer(self)

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

    @property
    def alpha_x(self):
        return self._kx * self.wavelength

    @property
    def alpha_y(self):
        return self._ky * self.wavelength

    @property
    def interpolation(self):
        return self._interpolation

    @property
    def aperture(self):
        return self._aperture

    @property
    def aberrations(self):
        return self._aberrations

    @property
    def translate(self):
        return self._translate

    @property
    def probe_gpts(self):
        return np.ceil(self.gpts / self.interpolation).astype(int)

    @property
    def position(self):
        return self._translate.position

    @position.setter
    def position(self, value):
        self._translate.position = value

    @property
    def k_max(self):
        return tf.reduce_max((tf.reduce_max(self._kx), tf.reduce_max(self._ky)))

    def get_probe(self):
        return self.get_tensor()

    def scan(self, scan, detectors=None, tracker=None):

        if detectors:
            scan._data = OrderedDict(zip(detectors, [[]] * len(detectors)))

        else:
            scan._data = []

        for i, position in enumerate(scan.generate_positions(1)):
            self.position = position[0]

            tensor = self.get_tensor()

            if detectors:
                for detector, detections in scan._data.items():
                    detections.append(detector.detect(tensor))

            else:
                scan._data.append(tensor)

        return scan

    def linescan(self, start, end, num_positions=None, sampling=None, endpoint=True, detectors=None):
        scan = LineScan(scanable=self, detectors=detectors, start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, detectors=detectors)

    def gridscan(self, start=None, end=None, num_positions=None, sampling=None, endpoint=False, detectors=None):

        scan = GridScan(scanable=self, detectors=detectors, start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, detectors=detectors)

    def get_coefficients(self):
        coefficients = self.aberrations.get_tensor()

        if self._aperture.radius != np.inf:
            coefficients *= tf.cast(self.aperture.get_tensor(), tf.complex64)

        return coefficients

    def _calculate_tensor(self):
        coefficients = self.get_coefficients()

        coefficients *= self.translate.get_tensor()

        begin = [0,
                 np.round((self.position[0] - self.extent[0] / (2 * self.interpolation)) /
                          self.sampling[0]).astype(int),
                 np.round((self.position[1] - self.extent[1] / (2 * self.interpolation)) /
                          self.sampling[1]).astype(int)]

        size = [self.kx.shape[0],
                np.ceil(self.gpts[0] / self.interpolation).astype(int),
                np.ceil(self.gpts[1] / self.interpolation).astype(int)]

        tensor = coefficients[:, None, None] * wrapped_slice(self._tensorflow, begin, size)

        return TensorWaves(tensor=tf.reduce_sum(tensor, axis=0, keepdims=True), extent=self.extent / self.interpolation,
                           energy=self.energy)

    def copy(self):
        return self.__class__(expansion=self._tensorflow, kx=self._kx, ky=self._ky, interpolation=self._interpolation,
                              extent=self.extent, energy=self.energy, save_tensor=self._save_tensor)
