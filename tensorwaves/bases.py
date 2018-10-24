import collections

import tensorflow as tf

import utils


def linspace_no_endpoint(n, l):
    return tf.lin_space(0., l - l / n, n)


def fftfreq(n, h):
    N = (n - 1) // 2 + 1
    p1 = tf.lin_space(0., N - 1, N)
    p2 = tf.lin_space(-float(n // 2), -1, n // 2)
    return tf.concat((p1, p2), axis=0) / (n * h)


def _consistent_grid(gpts, extent, sampling):
    if (gpts is not None) & (extent is not None):
        sampling = tuple(l / n for n, l in zip(gpts, extent))

    elif (gpts is not None) & (sampling is not None):
        extent = tuple(n * h for n, h in zip(gpts, sampling))

    elif (extent is not None) & (sampling is not None):
        gpts = tuple(int(l / h) for l, h in zip(extent, sampling))
        sampling = tuple(l / n for n, l in zip(gpts, extent))

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
        tokens.append('x'.join(map(str, gpts)) + ' gpts')
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


class Grid(object):

    def __init__(self, gpts=None, extent=None, sampling=None):
        self._gpts = gpts
        self._extent = extent
        self._sampling = sampling

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
        return tuple(linspace_no_endpoint(n, l) for n, l in zip(self.gpts, self.extent))

    def fftfreq(self):
        return tuple(fftfreq(n, h) for n, h in zip(self.gpts, self.sampling))

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

    def adapt(self, other):
        if self._energy is None:
            self._energy = other._energy

    def is_compatible(self, other):
        if (self._energy is not None) & (other._energy is not None):
            return self._energy == other._energy

    def __repr__(self):
        return _print_energy(self.energy)


class FactoryBase(Energy, Grid):

    def __init__(self, energy=None, gpts=None, extent=None, sampling=None):
        Energy.__init__(self, energy=energy)
        Grid.__init__(self, gpts=gpts, extent=extent, sampling=sampling)

    def adapt(self, other):
        Grid.adapt(self, other)
        Energy.adapt(self, other)

    def is_compatible(self, other):
        if not Grid.is_compatible(self, other):
            return False
        else:
            return Energy.is_compatible(self, other)

    def __repr__(self):
        return '{}\n{}'.format(Grid.__repr__(self), Energy.__repr__(self))


class TensorBase(Energy, Grid):

    def __init__(self, tensor, energy=None, extent=None, sampling=None, dimension=None):
        self._tensor = tensor

        self._dimension = None
        if isinstance(extent, collections.Iterable):
            self._dimension = len(extent)
        elif isinstance(sampling, collections.Iterable):
            self._dimension = len(sampling)

        if self._dimension is None:
            self._dimension = dimension

        if (self._dimension != dimension) & (dimension is not None):
            raise RuntimeError()

        if self._dimension is None:
            self._dimension = len(self.shape)

        Energy.__init__(self, energy=energy)
        Grid.__init__(self, extent=extent, sampling=sampling)

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
        Grid.adapt(self, other)
        Energy.adapt(self, other)

    def is_compatible(self, other):
        if not Grid.is_compatible(self, other):
            return False
        else:
            return Energy.is_compatible(self, other)

    def __repr__(self):
        return '{}\n{}'.format(Grid.__repr__(self), Energy.__repr__(self))
