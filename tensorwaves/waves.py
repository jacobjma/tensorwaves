from collections import Iterable
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from ase import Atoms

from tensorwaves.bases import TensorWithGridAndEnergy, HasGridAndEnergy, TensorFactory, notifying_property, GridProperty
from tensorwaves.bases import LineProfile, TensorWithGrid
from tensorwaves.potentials import Potential
from tensorwaves.scan import LineScan, GridScan, CustomScan
from tensorwaves.transfer import CTF, Translate, PrismTranslate, PrismCTF
from tensorwaves.utils import create_progress_bar, complex_exponential, fourier_propagator, fft_shift
from tensorwaves.analyse import fwhm


class TensorWaves(TensorWithGridAndEnergy):

    def __init__(self, tensor, extent=None, sampling=None, energy=None):

        tensor = tf.convert_to_tensor(tensor, dtype=tf.complex64)

        if len(tensor.shape) != 3:
            raise RuntimeError('')

        TensorWithGridAndEnergy.__init__(self, tensor, extent=extent, sampling=sampling, energy=energy, space='direct')

    def multislice(self, potential, in_place=False, progress_bar=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match_grid_and_energy(potential)

        self.check_is_defined()
        potential.check_is_defined()

        if in_place:
            wave = self
        else:
            wave = self.copy()

        propagator = self.get_fourier_propagator(potential.slice_thickness)

        # print(wave._tensor)
        # sss
        for potential_slice in create_progress_bar(potential.slice_generator(),
                                                   num_iter=potential.num_slices,
                                                   description='Multislice',
                                                   disable=not progress_bar):
            wave._tensor = wave._tensor * complex_exponential(wave.sigma * potential_slice._tensor)
            wave._tensor = tf.signal.ifft2d(tf.signal.fft2d(wave._tensor) * propagator)

        return wave

    def get_fourier_propagator(self, dz):
        kx, ky = self.fftfreq()
        return fourier_propagator(((kx ** 2)[:, None] + (ky ** 2)[None, :]), dz, self.wavelength)[None, :, :]

    def propagate(self, dz):
        self._tensor = tf.signal.ifft2d(tf.signal.fft2d(self.tensor()) * self.get_fourier_propagator(dz))

    def apply_ctf(self, in_place=False, **kwargs):
        ctf = CTF(**kwargs)

        return self.apply_frequency_transfer(ctf, in_place=in_place)

    def apply_frequency_transfer(self, frequency_transfer, in_place=False):

        self.match_grid_and_energy(frequency_transfer)

        self.check_is_defined()
        frequency_transfer.check_is_defined()

        frequency_transfer_tensor = frequency_transfer.build()

        tensor = tf.signal.ifft2d(tf.signal.fft2d(self.tensor()) * frequency_transfer_tensor.tensor())

        if in_place:
            self._tensor = tensor
            return self
        else:
            return TensorWaves(tensor, extent=self.extent.copy(), energy=self.energy)

    def diffractogram(self):
        from tensorwaves.image import Image
        return Image(fft_shift(tf.abs(tf.signal.fft2d(self.tensor())) ** 2, (1, 2)), extent=self.extent.copy())

    def intensity(self):
        new_tensor = TensorWithGrid(tensor=tf.abs(self.tensor()) ** 2, extent=self.extent)
        return new_tensor

    def copy(self):
        return self.__class__(tensor=tf.identity(self._tensor), extent=self.extent.copy(), energy=self.energy)


class WaveFactory(HasGridAndEnergy, TensorFactory):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        TensorFactory.__init__(self, save_tensor=save_tensor)
        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)

    def multislice(self, potential, progress_bar=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match_grid_and_energy(potential)

        return self.build().multislice(potential, in_place=True, progress_bar=progress_bar)


class PlaneWaves(WaveFactory):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None, save_tensor=True):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_tensor=save_tensor)

        self._num_waves = num_waves

    num_waves = notifying_property('_num_waves')

    def _calculate_tensor(self):
        tensor = tf.ones((self.num_waves, self.gpts[0], self.gpts[1]), dtype=tf.complex64)
        return TensorWaves(tensor, extent=self.extent, energy=self.energy)


class ProbeWaves(WaveFactory):

    def __init__(self, positions=None, aperture_cutoff=np.inf, aperture_rolloff=0., normalize=True, extent=None,
                 gpts=None, sampling=None, energy=None, save_tensor=False, **kwargs):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_tensor=save_tensor)

        self._ctf = CTF(aperture_cutoff=aperture_cutoff, aperture_rolloff=aperture_rolloff, save_tensor=save_tensor,
                        grid=self.grid, energy_wrapper=self.energy_wrapper, **kwargs)

        self._translate = Translate(positions=positions, save_tensor=save_tensor, grid=self.grid)

        self._normalize = normalize

        for depencency in self.get_dependencies():
            self.observe(depencency)

    def get_dependencies(self):
        dependencies = [self.grid, self.energy_wrapper, self.ctf, self.translate]
        dependencies += self.ctf.get_dependencies()
        return dependencies

    @property
    def ctf(self):
        return self._ctf

    @property
    def aberrations(self):
        return self.ctf.aberrations

    @property
    def aperture(self):
        return self.ctf.aperture

    @property
    def temporal_envelope(self):
        return self.ctf.temporal_envelope

    @property
    def gaussian_envelope(self):
        return self.ctf.gaussian_envelope

    @property
    def translate(self):
        return self._translate

    @property
    def positions(self):
        return self.translate.positions

    @positions.setter
    def positions(self, value):
        self.translate.positions = value

    def _calculate_tensor(self):

        tensor = self.ctf.build().tensor() * self.translate.build().tensor()

        if self._normalize:
            tensor /= tf.cast(tf.reduce_sum(tf.abs(tensor[0]) ** 2), tf.complex64)[None]

        tensor = tf.signal.fft2d(tensor)

        return TensorWaves(tensor, extent=self.extent, energy=self.energy)

    def scan(self, scan, potential, max_batch, detectors, progress_bar=True):

        if not isinstance(detectors, Iterable):
            detectors = [detectors]

        scan._data = OrderedDict(zip(detectors, [[]] * len(detectors)))

        for i, positions in enumerate(scan.generate_positions(max_batch, progress_bar=progress_bar)):
            self.positions = positions

            tensor = self.multislice(potential, progress_bar=False)

            for detector, detections in scan._data.items():
                detections.append(detector.detect(tensor))

        return scan

    def custom_scan(self, potential, max_batch, positions, detectors=None, progress_bar=True):
        scan = CustomScan(positions=positions)
        return self.scan(scan=scan, potential=potential, max_batch=max_batch, detectors=detectors,
                         progress_bar=progress_bar)

    def linescan(self, potential, max_batch, start, end, num_positions=None, sampling=None, endpoint=True,
                 detectors=None, progress_bar=True):
        scan = LineScan(start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, potential=potential, detectors=detectors, max_batch=max_batch,
                         progress_bar=progress_bar)

    def get_fwhm(self):
        old_positions = self.positions
        self.positions = self.extent / 2
        fwhm_value = fwhm(self.build().intensity().profile())
        self.positions = old_positions

        return fwhm_value

    def show_profile(self):
        old_positions = self.positions
        self.positions = self.extent / 2
        self.build().intensity().profile().show()
        self.positions = old_positions

    def gridscan(self, potential, max_batch, start=None, end=None, num_positions=None, sampling=None,
                 endpoint=False,
                 detectors=None):
        scan = GridScan(start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, potential=potential, max_batch=max_batch, detectors=detectors)


class PrismWaves(WaveFactory):

    def __init__(self, cutoff=0.01, interpolation=1., gpts=None, extent=None, sampling=None, energy=None,
                 save_tensor=True):
        self.cutoff = cutoff
        self.interpolation = interpolation

        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_tensor=save_tensor)

    def get_scattering_matrix(self):
        return self.build()

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

        tensor = complex_exponential(-2 * np.pi * (kx[:, None, None] * x[None, :, None] +
                                                   ky[:, None, None] * y[None, None, :]))

        return ScatteringMatrix(tensor, kx=kx, ky=ky, interpolation=self.interpolation, cutoff=self.cutoff,
                                extent=self.extent, energy=self.energy)


def wrapped_slice(tensor, begin, size):
    shift = [-x for x in begin]
    tensor = tf.roll(tensor, shift, list(range(len(begin))))
    return tf.slice(tensor, [0] * len(begin), size)


class ScatteringMatrix(HasGridAndEnergy, TensorFactory):

    def __init__(self, expansion, kx, ky, interpolation, cutoff, position=None, extent=None, sampling=None,
                 energy=None, save_tensor=True, **kwargs):

        if kx.shape != ky.shape:
            raise RuntimeError('')

        if expansion.shape[0] != kx.shape[0]:
            raise RuntimeError('')

        self._kx = kx
        self._ky = ky
        self._interpolation = interpolation
        self._cutoff = cutoff
        self._expansion = expansion

        TensorFactory.__init__(self, save_tensor=save_tensor)

        gpts = GridProperty(lambda: self.gpts, dtype=np.int32, locked=True)
        HasGridAndEnergy.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy)

        self._translate = PrismTranslate(position=position, kx=kx, ky=ky, save_tensor=True)
        self._ctf = PrismCTF(aperture_cutoff=np.inf, aperture_rolloff=0., focal_spread=0.,
                             gaussian_envelope_width=np.inf,
                             gaussian_envelope_scale=1., parametrization='polar', kx=kx, ky=ky,
                             save_tensor=save_tensor, energy_wrapper=self.energy_wrapper, **kwargs)

        for depencency in self.get_dependencies():
            self.observe(depencency)

    def get_dependencies(self):
        dependencies = [self.grid, self.energy_wrapper, self.ctf, self.translate]
        dependencies += self.ctf.get_dependencies()
        return dependencies

    @property
    def gpts(self):
        return np.array([dim for dim in self._expansion.shape[1:]])

    @property
    def cutoff(self):
        return self._cutoff

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
    def probe_gpts(self):
        return np.ceil(self.gpts / self.interpolation).astype(int)

    @property
    def probe_extent(self):
        return self.extent / self.interpolation

    @property
    def ctf(self):
        return self._ctf

    @property
    def aberrations(self):
        return self.ctf.aberrations

    @property
    def aperture(self):
        return self.ctf.aperture

    @property
    def gaussian_envelope(self):
        return self.ctf.gaussian_envelope

    @property
    def temporal_envelope(self):
        return self.ctf.temporal_envelope

    @property
    def translate(self):
        return self._translate

    @property
    def position(self):
        return self._translate.positions[0]

    @position.setter
    def position(self, value):
        self._translate.positions = value

    @property
    def k_max(self):
        return tf.reduce_max((tf.reduce_max(self._kx), tf.reduce_max(self._ky)))

    def multislice(self, potential, in_place=False, progress_bar=True):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.match_grid_and_energy(potential)

        self.check_is_defined()
        potential.check_is_defined()

        if in_place:
            S = self
        else:
            S = self.copy()

        propagator = self.get_fourier_propagator(potential.slice_thickness)

        for potential_slice in create_progress_bar(potential.slice_generator(),
                                                   num_iter=potential.num_slices,
                                                   description='Multislice',
                                                   disable=not progress_bar):
            S._expansion = S._expansion * complex_exponential(S.sigma * potential_slice._tensor)
            S._expansion = tf.signal.ifft2d(tf.signal.fft2d(S._expansion) * propagator)

        return S

    def get_fourier_propagator(self, dz):
        kx, ky = self.fftfreq()
        return fourier_propagator(((kx ** 2)[:, None] + (ky ** 2)[None, :]), dz, self.wavelength)[None, :, :]

    def get_probe(self):
        return self.build()

    def scan(self, scan, detectors, tracker=None):

        if not isinstance(detectors, Iterable):
            detectors = [detectors]

        scan._data = OrderedDict(zip(detectors, [[]] * len(detectors)))

        for i, position in enumerate(scan.generate_positions(1)):
            self.position = position[0]

            tensor = self.build()

            for detector, detections in scan._data.items():
                detections.append(detector.detect(tensor))

        return scan

    def linescan(self, start, end, num_positions=None, sampling=None, endpoint=True, detectors=None):
        scan = LineScan(scanable=self, start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, detectors=detectors)

    def gridscan(self, start=None, end=None, num_positions=None, sampling=None, endpoint=False, detectors=None):

        scan = GridScan(scanable=self, start=start, end=end, num_positions=num_positions,
                        sampling=sampling, endpoint=endpoint)

        return self.scan(scan=scan, detectors=detectors)

    def custom_scan(self, positions, detectors=None):
        scan = CustomScan(scanable=self, positions=positions)
        return self.scan(scan, detectors=detectors)

    def _calculate_tensor(self):

        begin = [0,
                 np.round((self.position[0] - self.extent[0] / (2 * self.interpolation)) /
                          self.sampling[0]).astype(int),
                 np.round((self.position[1] - self.extent[1] / (2 * self.interpolation)) /
                          self.sampling[1]).astype(int)]

        size = [self.kx.shape[0],
                np.ceil(self.gpts[0] / self.interpolation).astype(int),
                np.ceil(self.gpts[1] / self.interpolation).astype(int)]

        coefficients = self.ctf.build()

        coefficients *= self.translate.build()  # .tensor()

        tensor = coefficients[:, None, None] * wrapped_slice(self._expansion, begin, size)

        return TensorWaves(tensor=tf.reduce_sum(tensor, axis=0, keepdims=True), extent=self.extent / self.interpolation,
                           energy=self.energy)

    def copy(self):
        S = self.__class__(expansion=self._expansion, kx=self._kx, ky=self._ky, interpolation=self._interpolation,
                           cutoff=self.cutoff, extent=self.extent, energy=self.energy, save_tensor=self._save_tensor)
        S._ctf = self.ctf.copy()
        S._translate = self.translate.copy()
        S.observe(S._ctf)
        S.observe(S._translate)
        return S
