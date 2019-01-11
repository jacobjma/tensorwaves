import numpy as np
import tensorflow as tf

from tensorwaves.bases import HasData, Showable, HasAccelerator, notifying_property, TensorWaves, Observer, Observable
from tensorwaves.potentials import Potential
from tensorwaves.scan import LineScan, GridScan
from tensorwaves.transfer import CTF, Translate, PrismAberration, PrismTranslate
from tensorwaves.utils import complex_exponential, wrapped_slice, freq2angles, bar
from ase import Atoms


class WaveFactory(HasData, Showable, HasAccelerator):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_wave=True, grid=None, accelerator=None):
        HasData.__init__(self, save_data=save_wave)
        Showable.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid, space='direct')
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        self.grid.register_observer(self)
        self.accelerator.register_observer(self)

    def multislice(self, potential, in_place=False):
        if isinstance(potential, Atoms):
            potential = Potential(atoms=potential)

        self.grid.match(potential.grid)

        wave = self.get_tensor()
        wave = wave.multislice(potential, in_place=in_place)
        return wave

    def check_is_defined(self):
        self._grid.check_is_defined()
        self._accelerator.check_is_defined()

    def get_semiangles(self, return_squared_norm=False, return_azimuth=False):
        return freq2angles(*self._grid.fftfreq(), self._accelerator.wavelength, return_squared_norm, return_azimuth)

    def get_tensor(self):
        return self.get_data()

    def tensorflow(self):
        return self.get_tensor().tensorflow()

    def numpy(self):
        return self.get_tensor().numpy()

    def get_showable_tensor(self, i=None):
        return self.get_data()

    # def show(self, space='direct', mode='magnitude', **kwargs):
    #    self.get_tensor().show(space=space, mode=mode, **kwargs)


class Node(object):

    def __init__(self, in_wave, operation, save_wave=True):
        self._in_wave = in_wave
        self._operation = operation
        self._out_wave = None
        self._up_to_date = False

    def get_tensor(self):
        if not self._up_to_date:
            wave = self._in_wave.get_tensor()

            if isinstance(self._operation, Potential):
                wave = wave.multislice(self._operation, in_place=False)
            else:
                wave = self._operation.apply(wave)

            self._up_to_date = True
            self._out_wave = wave
        else:
            wave = self._out_wave

        return wave


class Pipeline(Observer, Observable):

    def __init__(self, operations):
        Observer.__init__(self)
        Observable.__init__(self)

        self._operations = operations

        for operation in operations:
            operation.register_observer(self)

        self._nodes = []
        self._nodes.append(operations[0])
        for i, operation in enumerate(operations[1:]):
            self._nodes.append(Node(self._nodes[i], operation))

        self._updating = False

    def notify(self, observable, message):
        if not self._updating:
            for node in self._nodes[self._operations.index(observable):]:
                node._up_to_date = False

            self.notify_observers(message)

    def get_tensor(self):
        self._updating = True

        tensor = self._nodes[-1].get_tensor()
        self._updating = False
        return tensor

    def _get_show_data(self):

        return self.get_tensor().numpy()


class PlaneWaves(WaveFactory):

    def __init__(self, num_waves=1, extent=None, gpts=None, sampling=None, energy=None, save_wave=True):
        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_wave=save_wave)

        self._num_waves = num_waves

    num_waves = notifying_property('_num_waves')

    def _calculate_data(self):
        self.check_is_defined()
        return TensorWaves(tf.ones((self.num_waves, self.grid.gpts[0], self.grid.gpts[1]), dtype=tf.complex64),
                           extent=self._grid.extent, energy=self._accelerator.energy)


class Scanable(object):

    def linescan(self, start, end, num_positions=None, sampling=None, endpoint=True, max_batch=1, potential=None,
                 detectors=None):
        scan = LineScan(start=start, end=end, num_positions=num_positions, sampling=sampling, endpoint=endpoint)

        return scan.scan(scan, max_batch=max_batch, potential=potential, detectors=detectors)

    def gridscan(self, start, end, num_positions=None, sampling=None, endpoint=True, max_batch=1, potential=None,
                 detectors=None):
        scan = GridScan(start=start, end=end, num_positions=num_positions, sampling=sampling, endpoint=endpoint)

        return scan.scan(scan, max_batch=max_batch, potential=potential, detectors=detectors)


class ProbeWaves(WaveFactory, Scanable):

    def __init__(self, positions=None, aperture_radius=1, aperture_rolloff=0., extent=None, gpts=None, sampling=None,
                 energy=None, save_wave=True, grid=None, accelerator=None, **kwargs):
        self._ctf = CTF(extent=extent, gpts=gpts, sampling=sampling, energy=energy, aperture_radius=aperture_radius,
                        aperture_rolloff=aperture_rolloff, **kwargs)

        self._translate = Translate(positions=positions, grid=self._ctf.grid, accelerator=self._ctf.accelerator)

        self._ctf.register_observer(self)
        self._ctf.aberrations.register_observer(self)
        self._ctf.aperture.register_observer(self)
        self._ctf.temporal_envelope.register_observer(self)

        self._translate.register_observer(self)

        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_wave=save_wave,
                             grid=self._ctf.grid, accelerator=self._ctf.accelerator)

        self._observing = self._ctf._observing + self._translate._observing + [self._translate]

    @property
    def ctf(self):
        return self._ctf

    @property
    def translate(self):
        return self._translate

    def _calculate_data(self):
        return TensorWaves(tf.fft2d(self._ctf.get_data()._tensor * self._translate.get_data()._tensor),
                           extent=self._grid.extent, energy=self._accelerator.energy)

    def generate_scan_positions(self, scan, max_batch=1):
        for positions in scan.generate_positions(max_batch):
            self.translate.positions = positions
            yield self.get_tensor()

    # def transmit(self, position, potential):
    #     self.match_grid(potential)
    #     tensor = self.get_tensor(position)
    #     tensor.multislice(potential)
    #     return tensor


class PrismWaves(WaveFactory):

    def __init__(self, cutoff, interpolation=1., gpts=None, extent=None, sampling=None, energy=None, save_wave=True,
                 grid=None, accelerator=None):
        self.cutoff = cutoff
        self.interpolation = interpolation

        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_wave=save_wave,
                             grid=grid, accelerator=accelerator)

    def get_scattering_matrix(self):
        return self.get_data()

    def _calculate_data(self):
        n_max = np.ceil(self.cutoff / (self.accelerator.wavelength / self.grid.extent[0] * self.interpolation))
        m_max = np.ceil(self.cutoff / (self.accelerator.wavelength / self.grid.extent[1] * self.interpolation))

        kx = tf.cast(tf.range(-n_max, n_max + 1), tf.float32) / self.grid.extent[0] * self.interpolation
        ky = tf.cast(tf.range(-m_max, m_max + 1), tf.float32) / self.grid.extent[1] * self.interpolation

        mask = tf.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) < (self.cutoff / self.accelerator.wavelength)

        ky, kx = tf.meshgrid(ky, kx)
        kx = tf.boolean_mask(kx, mask)
        ky = tf.boolean_mask(ky, mask)

        x, y = self.grid.linspace()

        tensor = complex_exponential(2 * np.pi * (kx[:, None, None] * x[None, :, None] +
                                                  ky[:, None, None] * y[None, None, :]))

        return ScatteringMatrix(tensor, kx=kx, ky=ky, interpolation=self.interpolation, extent=self.grid.extent,
                                energy=self.accelerator.energy)

    # def show_probe(self, x=None, y=None, mode='magnitude', display_space='direct', ax=None, defocus=0., **kwargs):
    #    self.get_tensor().show_probe(x=x, y=y, mode=mode, display_space=display_space, ax=ax, defocus=defocus, **kwargs)

    # def generate(self, max_waves):
    #     x, y, kx, ky = self._coordinates()
    #
    #     for start, size in batch_generator(kx.shape[0].value, max_waves):
    #         end = start + size
    #         yield ScatteringMatrix(tensor=self._tensor(x, y, kx[start:end], kx[start:end]), kx=kx[start:end],
    #                                ky=ky[start:end],
    #                                interpolation_factor=self._interpolation_factor,
    #                                energy=self.energy, extent=self.extent, sampling=self.sampling)
    #
    # def show_probe(self, mode='abs2', space='direct', **kwargs):
    #     self.build().show_probe(mode, space, **kwargs)


class ScatteringMatrix(TensorWaves, HasData, Scanable):

    def __init__(self, tensor, kx, ky, interpolation, position=None, extent=None, sampling=None, grid=None, energy=None,
                 accelerator=None, save_wave=True, **kwargs):
        self._kx = kx
        self._ky = ky
        self._interpolation = interpolation

        TensorWaves.__init__(self, tensor=tensor, extent=extent, sampling=sampling, grid=grid, energy=energy,
                             accelerator=accelerator)
        HasData.__init__(self, save_data=save_wave)

        self._aberrations = PrismAberration(kx=kx, ky=ky, accelerator=self.accelerator, **kwargs)
        self._translate = PrismTranslate(kx=-kx, ky=-ky, positions=position)

        self._translate.register_observer(self)
        self._aberrations.register_observer(self)

        self._observing += [self._aberrations, self._translate]

    @property
    def position(self):
        return self._translate.positions[0]

    @position.setter
    def position(self, value):
        self._translate.positions = value

    @property
    def translate(self):
        return self._translate

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

    @property
    def interpolation(self):
        return self._interpolation

    def get_tensor(self):
        return self.get_data()

    def get_probe(self):
        return self.get_data()

    def get_showable_tensor(self):
        return self.get_data()

    def _calculate_data(self):
        tensor = self._aberrations.get_data()

        tensor *= self._translate._calculate_data()

        begin = [0,
                 np.round((self.position[0] - self.grid.extent[0] / (2 * self.interpolation)) /
                          self.grid.sampling[0]).astype(int),
                 np.round((self.position[1] - self.grid.extent[1] / (2 * self.interpolation)) /
                          self.grid.sampling[1]).astype(int)]

        size = [self.kx.shape[0].value,
                np.ceil(self.grid.gpts[0] / self.interpolation).astype(int),
                np.ceil(self.grid.gpts[1] / self.interpolation).astype(int)]

        tensor = tensor[:, None, None] * wrapped_slice(self._tensor, begin, size)

        return TensorWaves(tensor=tf.reduce_sum(tensor, axis=0, keep_dims=True),
                           extent=self._grid.extent / self.interpolation, energy=self._accelerator.energy)

    def copy(self):
        return self.__class__(tensor=self._tensor, kx=self._kx, ky=self._ky, interpolation=self._interpolation)

    # def generate_scan_positions(self, scan, max_batch=1):
    #     for positions in scan.generate_positions(1):
    #         self.translate.positions = positions
    #         yield self.get_tensor()
