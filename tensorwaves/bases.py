import numpy as np
from tensorwaves.utils import energy2wavelength, energy2sigma, linspace_no_endpoint, fftfreq


class Grid(object):

    def __init__(self, extent=None, gpts=None, sampling=None, dimension=2, adjust=None):
        self._extent = None
        self._gpts = None
        self._sampling = None
        self._dimension = dimension

        if adjust is None:
            self._adjust = {'extent': 'gpts', 'gpts': 'sampling', 'sampling': 'gpts'}
        else:
            self._adjust = adjust

        if extent is not None:
            self.extent = extent

        if gpts is not None:
            self.gpts = gpts

        if sampling is not None:
            self.sampling = sampling

    def _get_grid_property(self, name):
        value = getattr(self, name)
        if (self._dimension == 1) & (value is not None):
            return value
        else:
            return value

    def _set_grid_property(self, name, value, adjust, dtype):
        if isinstance(value, np.ndarray):
            setattr(self, name, value.astype(dtype))
        else:
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(self._dimension, dtype=dtype))

            getattr(self, name)[:] = value

        self._adjust_consistent(adjust)

    @property
    def extent(self):
        return self._get_grid_property('_extent')

    @extent.setter
    def extent(self, value):
        self._set_grid_property('_extent', value, adjust=self._adjust['extent'], dtype=np.float32)

    @property
    def gpts(self):
        return self._get_grid_property('_gpts')

    @gpts.setter
    def gpts(self, value):
        self._set_grid_property('_gpts', value, adjust=self._adjust['gpts'], dtype=np.int32)

    @property
    def sampling(self):
        return self._get_grid_property('_sampling')

    @sampling.setter
    def sampling(self, value):
        self._set_grid_property('_sampling', value, adjust=self._adjust['sampling'], dtype=np.float32)

    def linspace(self):
        value = []
        for n, l in zip(self.gpts, self.extent):
            value += [linspace_no_endpoint(n, l)]
        return value

    def fftfreq(self):
        value = []
        for n, h in zip(self.gpts, self.sampling):
            value += [fftfreq(n, h)]
        return value

    def _adjust_consistent(self, adjust):
        if adjust.lower() == 'extent':
            if (self.gpts is not None) & (self.sampling is not None):
                if self.extent is None:
                    self._extent = np.zeros(self._dimension, dtype=np.float32)
                self._extent[:] = self.sampling * self.gpts

        elif adjust.lower() == 'gpts':
            if (self.extent is not None) & (self.sampling is not None):
                if self.gpts is None:
                    self._gpts = np.zeros(self._dimension, dtype=np.int)
                self._gpts[:] = (np.ceil(self.extent / self.sampling)).astype(np.int)
                self._sampling[:] = self.extent / self.gpts

        elif adjust.lower() == 'sampling':
            if (self.extent is not None) & (self.gpts is not None):
                if self.sampling is None:
                    self._sampling = np.zeros(self._dimension, dtype=np.float32)
                self._sampling[:] = (self.extent / self.gpts).astype(np.float32)

    def copy(self):
        return self.__class__(self.extent.copy(), self.gpts.copy(), self.sampling.copy(), self._dimension,
                              self._adjust.copy())


class Accelerator(object):

    def __init__(self, energy):
        self.energy = energy

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    @property
    def interaction_parameter(self):
        return energy2sigma(self.energy)

    def copy(self):
        return self.__class__(self.energy)


class TensorFactory(object):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None):
        self.grid = Grid(extent=extent, gpts=gpts, sampling=sampling, dimension=2)
        self.accelerator = Accelerator(energy=energy)

    def adopt_grid(self, other):
        self.grid.extent = other.grid.extent.copy()
        self.grid.gpts = other.grid.gpts.copy()

    def adopt_energy(self, other):
        self.accelerator.energy = other.accelerator.energy
