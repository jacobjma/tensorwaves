from numbers import Number

import numpy as np
import tensorflow as tf
from ase import Atoms

from tensorwaves.bases import HasData, HasGrid, HasAccelerator, notifying_property, TensorWaves, Observer, Observable, \
    GridException
from tensorwaves.plotutils import show_array
from tensorwaves.transfer import CTF, Translate
from tensorwaves.utils import complex_exponential, fft_shift, wrapped_slice, freq2angles, bar
from tensorwaves.scan import LineScan, GridScan
from tensorwaves.potentials import Potential


class WaveFactory(HasData, HasGrid, HasAccelerator):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_wave=True, grid=None, accelerator=None):
        HasData.__init__(self, save_data=save_wave)
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, space='direct', grid=grid)
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        self.grid.register_observer(self)
        self.accelerator.register_observer(self)

    def multislice(self, potential, in_place=False):
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

    def _get_show_data(self):
        return self.numpy()

    def show(self, space='direct', mode='magnitude', **kwargs):
        self.get_tensor().show(space=space, mode=mode, **kwargs)


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


# class Source(object):
#
#     def __init__(self, in_wave):

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


class ProbeWaves(WaveFactory):

    def __init__(self, positions=None, aperture_radius=1, aperture_rolloff=0., extent=None, gpts=None, sampling=None,
                 energy=None, save_wave=True, grid=None, accelerator=None, **kwargs):
        self._ctf = CTF(extent=extent, gpts=gpts, sampling=sampling, energy=energy, aperture_radius=aperture_radius,
                        aperture_rolloff=aperture_rolloff, **kwargs)

        self._translate = Translate(positions=positions, grid=self._ctf.grid, accelerator=self._ctf.accelerator)

        self._ctf.register_observer(self)
        self._translate.register_observer(self)

        WaveFactory.__init__(self, extent=extent, gpts=gpts, sampling=sampling, energy=energy, save_wave=save_wave,
                             grid=self._ctf.grid, accelerator=self._ctf.accelerator)

    @property
    def ctf(self):
        return self._ctf

    @property
    def translate(self):
        return self._translate

    def _calculate_data(self):
        return TensorWaves(tf.fft2d(self._ctf.get_data() * self._translate.get_data()), extent=self._grid.extent,
                           energy=self._accelerator.energy)

    # def transmit(self, position, potential):
    #     self.match_grid(potential)
    #     tensor = self.get_tensor(position)
    #     tensor.multislice(potential)
    #     return tensor

    def linescan(self, start, end, num_positions=None, sampling=None, endpoint=True, max_batch=1, potential=None,
                 detectors=None):

        scan = LineScan(start=start, end=end, num_positions=num_positions, sampling=sampling, endpoint=endpoint)

        return self.scan(scan, max_batch=max_batch, potential=potential, detectors=detectors)

    def scan(self, scan, max_batch=1, potential=None, detectors=None):

        for detector in detectors:
            scan.register_detector(detector)

        num_iter = (scan.num_positions + max_batch - 1) // max_batch

        for tensor in bar(self.generate_scan_positions(scan, max_batch), num_iter):
            tensor = tensor.multislice(potential)
            scan.detect(tensor)

        return scan

    def generate_scan_positions(self, scan, max_batch):
        for positions in scan.generate_positions(max_batch):
            self.translate.positions = positions
            yield self.get_tensor()

    # def grid_scan(self, max_positions_per_batch, origin=None, extent=None, num_positions=None, sampling=None,
    #               potential=None, detector=None):
    #     if origin is None:
    #         origin = np.zeros(2, dtype=np.float32)
    #
    #     if self.extent is None:
    #         self.extent = potential.extent
    #
    #     if extent is None:
    #         extent = potential.extent
    #
    #     potential.sampling = self.sampling
    #
    #     scan = GridScan(origin, extent, num_positions)
    #
    #     for positions in scan.generate_positions(max_positions_per_batch):
    #         tensor = self.get_tensor(positions)
    #         tensor.multislice(potential)
    #         yield tensor
    #
    # def show(self, mode='magnitude', display_space='direct', axes=None, **kwargs):
    #     self.get_tensor().show(mode=mode, display_space=display_space, **kwargs)


class PrismWaves(WaveFactory):

    def __init__(self, cutoff, interpolation=1., gpts=None, extent=None, sampling=None, energy=None):
        self.cutoff = cutoff
        self.interpolation = interpolation

        WaveFactory.__init__(self, gpts=gpts, extent=extent, sampling=sampling, energy=energy)

    def _create_tensor(self, i):
        n_max = np.ceil(self.cutoff / (self.wavelength / self.extent[0] * self.interpolation))
        m_max = np.ceil(self.cutoff / (self.wavelength / self.extent[1] * self.interpolation))

        kx = tf.cast(tf.range(-n_max, n_max + 1), tf.float32) / self.extent[0] * self.interpolation
        ky = tf.cast(tf.range(-m_max, m_max + 1), tf.float32) / self.extent[1] * self.interpolation

        mask = tf.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) < (self.cutoff / self.wavelength)

        ky, kx = tf.meshgrid(ky, kx)
        kx = tf.boolean_mask(kx, mask)
        ky = tf.boolean_mask(ky, mask)

        x, y = self.linspace()

        if i is None:
            tensor = complex_exponential(2 * np.pi * (kx[:, None, None] * x[None, :, None] +
                                                      ky[:, None, None] * y[None, None, :]))

            return tensor, kx, ky
        else:
            tensor = complex_exponential(2 * np.pi * (kx[i, None, None] * x[None, :, None] +
                                                      ky[i, None, None] * y[None, None, :]))

            return tensor, kx[i, None], ky[i, None]

    def get_tensor(self, i=None):
        tensor, kx, ky = self._create_tensor(i=i)

        return ScatteringMatrix(tensor, kx=kx, ky=ky, interpolation=self.interpolation, extent=self.extent,
                                energy=self.energy)

    def show_probe(self, x=None, y=None, mode='magnitude', display_space='direct', ax=None, defocus=0., **kwargs):
        self.get_tensor().show_probe(x=x, y=y, mode=mode, display_space=display_space, ax=ax, defocus=defocus, **kwargs)

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


class ScatteringMatrix(TensorWaves):

    def __init__(self, tensor, kx, ky, interpolation, extent=None, sampling=None, energy=None):
        self._kx = kx
        self._ky = ky
        self._interpolation = interpolation

        TensorWaves.__init__(self, tensor=tensor, extent=extent, sampling=sampling, energy=energy)

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky

    @property
    def interpolation(self):
        return self._interpolation

    def probe(self, x, y, parametrization='polar', **kwargs):
        ctf = CTF(energy=self.energy, parametrization=parametrization, **kwargs)

        alpha_x = self.kx * self.wavelength
        alpha_y = self.ky * self.wavelength

        alpha2 = alpha_x ** 2 + alpha_y ** 2
        alpha = tf.sqrt(alpha2)

        phi = tf.atan2(alpha_x, alpha_y)

        chi = 2 * np.pi / self.wavelength * ctf.parametrization.get_function()(alpha, alpha ** 2, phi)

        coefficients = complex_exponential(
            -2 * np.pi * (self.kx * x + self.ky * y) - chi)[:, None, None]

        begin = [0,
                 np.round((x - self.extent[0] / (2 * self.interpolation)) / self.sampling[0]).astype(int),
                 np.round((y - self.extent[1] / (2 * self.interpolation)) / self.sampling[1]).astype(int)]

        size = [self.kx.shape[0].value,
                np.ceil(self.gpts[0] / self.interpolation).astype(int),
                np.ceil(self.gpts[1] / self.interpolation).astype(int)]

        tensor = wrapped_slice(self._tensor, begin, size)

        return tf.reduce_sum(tensor * coefficients, axis=0)

    def show_probe(self, x=None, y=None, mode='magnitude', display_space='direct', ax=None, defocus=0., **kwargs):
        if x is None:
            x = self.extent[0] / 2

        if y is None:
            y = self.extent[1] / 2

        tensor = self.probe(x, y, defocus=defocus)
        show_array(tensor.numpy().T, self.space, self.extent, mode=mode, display_space=display_space, ax=ax, **kwargs)
