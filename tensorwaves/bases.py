import collections

import tensorflow as tf
import numpy as np

from tensorwaves import utils


def _consistent_grid(gpts, extent, sampling):
    if (gpts is not None) & (extent is not None):
        sampling = tf.constant([l / n for n, l in zip(gpts, extent)], dtype=tf.float32)

    elif (gpts is not None) & (sampling is not None):
        extent = tf.constant([n * h for n, h in zip(gpts, sampling)], dtype=tf.float32)

    elif (extent is not None) & (sampling is not None):
        gpts = tf.ceil(tf.constant(extent, dtype=tf.float32) / sampling)
        sampling = tf.constant(extent, dtype=tf.float32) / gpts
        gpts = tf.cast(gpts, dtype=tf.int32)

    return gpts, extent, sampling


def _is_grid_compatible(grid1, grid2):
    if (grid1.gpts is not None) & (grid2.gpts is not None):
        if grid1.gpts != grid2.gpts:
            return False

    if (grid1.extent is not None) & (grid2.extent is not None):
        if grid1.extent != grid2.extent:
            return False

    if (grid1.sampling is not None) & (grid2.sampling is not None):
        if grid1.sampling != grid2.sampling:
            return False

    return True


def _print_grid(gpts, extent, sampling):
    tokens = []
    if extent is not None:
        tokens.append('x'.join(map(lambda x: ('%.6f' % x).rstrip('0').rstrip('.'), extent)) + ' Angstrom')
    else:
        tokens.append('no scale')

    if gpts is not None:
        tokens.append('x'.join(map(lambda x: '%d' % x, gpts)) + ' gpts')
    else:
        tokens.append('no grid')

    if sampling is not None:
        tokens.append(
            'x'.join(map(lambda x: ('%.6f' % x).rstrip('0').rstrip('.'), sampling)) + ' Angstrom / gpt')
    else:
        tokens.append('no resolution')

    return '{}, {} ({})'.format(*tokens)


def _print_energy(energy):
    if energy is None:
        return 'no energy'
    else:
        wavelength = utils.energy2wavelength(energy)
        return '{} keV ({} Angstrom)'.format(energy / 1000, ('%.6f' % wavelength).rstrip('0').rstrip('.'))


class ZAxis(object):

    def __init__(self, entrance_plane=0, exit_plane=None):
        self._entrance_plane = entrance_plane
        self._exit_plane = exit_plane

    @property
    def depth(self):
        if (self.exit_plane is None) | (self.entrance_plane is None):
            return None
        else:
            return self._exit_plane - self._entrance_plane

    @property
    def entrance_plane(self):
        return self._entrance_plane

    @property
    def exit_plane(self):
        return self._exit_plane


class Box(ZAxis):

    def __init__(self, extent=None, entrance_plane=0, exit_plane=None):
        self._extent = extent
        super().__init__(entrance_plane=entrance_plane, exit_plane=exit_plane)

    @property
    def box(self):
        if (self.extent is None) | (self.depth is None):
            return None
        else:
            return tf.concat((self.extent, [self.depth]), 0)

    @property
    def extent(self):
        return self._extent

    @property
    def gpts(self):
        return None

    @property
    def sampling(self):
        return None

    def is_compatible(self, other):
        return self._extent == other.extent

    def adapt(self, other):
        self._extent = other.extent


class XYGrid(object):

    def __init__(self, gpts=None, extent=None, sampling=None):
        if gpts is not None:
            self._gpts = tf.constant(gpts, tf.int32)
        else:
            self._gpts = None
        if extent is not None:
            self._extent = tf.constant(extent, tf.float32)
        else:
            self._extent = None
        if sampling is not None:
            self._sampling = tf.constant(sampling, tf.float32)
        else:
            self._sampling = None

        self._set_consistent_grid(self.gpts, self.extent, self.sampling)

    def _set_consistent_grid(self, gpts, extent, sampling):
        gpts, extent, sampling = _consistent_grid(gpts, extent, sampling)
        self._gpts = gpts
        self._extent = extent
        self._sampling = sampling

    @property
    def gpts(self):
        return self._gpts

    @gpts.setter
    def gpts(self, gpts):
        self._set_consistent_grid(gpts, self.extent, self.sampling)

    @property
    def extent(self):
        return self._extent

    @extent.setter
    def extent(self, extent):
        self._set_consistent_grid(self.gpts, extent, self.sampling)

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self, sampling):
        if self.extent is None:
            gpts = self.gpts
        else:
            gpts = None
        self._set_consistent_grid(gpts, self.extent, sampling)

    def is_compatible(self, other):
        return _is_grid_compatible(self, other)

    def adapt(self, other):
        if not _is_grid_compatible(self, other):
            raise RuntimeError('incompatible grids')

        gpts, extent, sampling = _consistent_grid(other.gpts, other.extent, other.sampling)

        if self.gpts is not None:
            gpts = self.gpts

        if self.extent is not None:
            extent = self.extent

        if self.sampling is not None:
            sampling = self.sampling

        self._set_consistent_grid(gpts, extent, sampling)

    def linspace(self):
        return tuple(utils.linspace_no_endpoint(n, l) for n, l in zip(self.gpts, self.extent))

    def fftfreq(self):
        return tuple(utils.fftfreq(n, h) for n, h in zip(self.gpts, self.sampling))

    def __repr__(self):
        return _print_grid(self.gpts, self.extent, self.sampling)


class Energy(object):

    def __init__(self, energy=None):
        self._energy = energy

    def _is_energy_defined(self):
        if self._energy is None:
            raise RuntimeError('the energy is not defined')

    @property
    def energy(self):
        return self._energy

    @property
    def wavelength(self):
        self._is_energy_defined()
        return utils.energy2wavelength(self._energy)

    @property
    def interaction_parameter(self):
        self._is_energy_defined()
        return utils.energy2sigma(self._energy)

    def adapt(self, other):
        if self._energy is None:
            self._energy = other._energy

    def is_compatible(self, other):
        if (self._energy is not None) & (other._energy is not None):
            return self._energy == other._energy

    def __repr__(self):
        return _print_energy(self.energy)


class FactoryBase(Energy, XYGrid):

    def __init__(self, energy=None, gpts=None, extent=None, sampling=None):
        Energy.__init__(self, energy=energy)
        XYGrid.__init__(self, gpts=gpts, extent=extent, sampling=sampling)

    def adapt(self, other):
        XYGrid.adapt(self, other)
        Energy.adapt(self, other)

    def is_compatible(self, other):
        if not XYGrid.is_compatible(self, other):
            return False
        else:
            return Energy.is_compatible(self, other)

    def __repr__(self):
        return '{}\n{}'.format(XYGrid.__repr__(self), Energy.__repr__(self))


class TensorBase(Energy, XYGrid):

    def __init__(self, tensor, energy=None, extent=None, sampling=None, dimension=None):
        self._tensor = tensor

        Energy.__init__(self, energy=energy)
        XYGrid.__init__(self, extent=extent, sampling=sampling, dimension=dimension)

        self._gpts = None

    @property
    def gpts(self):
        return self.shape[-self._dimension:]

    @gpts.setter
    def gpts(self, _):
        raise RuntimeError()

    @property
    def shape(self):
        return tuple(dim.value for dim in self._tensor.get_shape())

    @property
    def tensor(self):
        return self._tensor

    def adapt(self, other):
        XYGrid.adapt(self, other)
        Energy.adapt(self, other)

    def is_compatible(self, other):
        if not XYGrid.is_compatible(self, other):
            return False
        else:
            return Energy.is_compatible(self, other)

    def __repr__(self):
        return '{}\n{}'.format(XYGrid.__repr__(self), Energy.__repr__(self))
