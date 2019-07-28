import numpy as np
import tensorflow as tf
from ase import units

from tensorwaves.plotutils import show_array, show_line


def named_property(name):
    def getter(self):
        return getattr(self, name)

    def setter(self, value):
        setattr(self, name, value)

    return property(getter, setter)


def referenced_property(reference_name, property_name):
    def getter(self):
        return getattr(getattr(self, reference_name), property_name)

    def setter(self, value):
        setattr(getattr(self, reference_name), property_name, value)

    return property(getter, setter)


def notifying_property(name):
    def getter(self):
        return getattr(self, name)

    def setter(self, value):
        old = getattr(self, name)
        setattr(self, name, value)
        change = np.all(old != value)
        self.notify_observers({'name': name, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


class Observable(object):
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.notify(self, message)


class Observer(object):
    def __init__(self, observable=None):
        self._observing = []

        if observable:
            self.observe(observable)

    def observe(self, observable):
        observable.register_observer(self)

    def notify(self, observable, message):
        raise NotImplementedError()


class TensorFactory(Observer):

    def __init__(self, save_tensor=True):
        Observer.__init__(self)
        self.up_to_date = False
        self._save_tensor = save_tensor
        self._tensor = None

    @property
    def save_tensor(self):
        return self._save_tensor

    def notify(self, observable, message):
        if message['change']:
            self.up_to_date = False

    def check_is_defined(self):
        raise NotImplementedError()

    def _calculate_tensor(self, *args):
        raise NotImplementedError()

    def build(self, *args):
        self.check_is_defined()

        if self.up_to_date & self._save_tensor:
            tensor = self._tensor
        else:
            tensor = self._calculate_tensor(*args)
            if self._save_tensor:
                self._tensor = tensor
                self.up_to_date = True
        return tensor

    def tensor(self):
        self.build()
        return self._tensor

    def numpy(self):
        return self.build()._tensor.numpy()

    def clear(self):
        self._tensor = None
        self.up_to_date = False


def linspace_no_endpoint(start, stop, num, dtype=tf.float32):
    """
    Return evenly spaced numbers over a specified half-open interval.

    Tensorflow version of numpy's linspace with endpoint set to false.

    Parameters
    ----------
    start : scalar
        The starting value of the interval.
    stop : scalar
        The end value of the interval.
    num : int
        Number of samples to generate.
    dtype : dtype, optional
        The type of the output tensor. Defaults to tf.float32.

    Returns
    -------
    samples : tensor
        There are num equally spaced samples in the half-open interval [start, stop).

    """
    start = tf.cast(start, dtype=dtype)
    interval = stop - start
    return tf.linspace(start, interval - interval / tf.cast(num, tf.float32), num)


def fftfreq(n, d=1.):
    """
    Return the Discrete Fourier Transform sample frequencies.

    Tensorflow version of Numpy's fftfreq.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : tensor
        Tensor of length n containing the sample frequencies.
    """
    m = (n - 1) // 2 + 1
    p1 = tf.linspace(0., m - 1, m)
    p2 = tf.linspace(-tf.cast(n // 2, tf.float32), -1, n // 2)
    return tf.concat((p1, p2), axis=0) / (n * d)


def fftfreq_range(extent, n):
    return np.array([-n / extent, n / extent]) / 2.


class GridProperty(object):

    def __init__(self, value, dtype, locked=False, dimensions=2):
        self._dtype = dtype
        self._locked = locked
        self._dimensions = dimensions
        self._value = self._validate(value)

    @property
    def locked(self):
        return self._locked

    @property
    def value(self):
        if self._locked:
            return self._validate(self._value())
        else:
            return self._value

    def _validate(self, value):
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self._dimensions:
                raise RuntimeError('grid value length of {} != {}'.format(len(value), self._dimensions))
            value = np.array(value).astype(self._dtype)

        elif isinstance(value, (int, float, complex)):
            value = np.full(self._dimensions, value, dtype=self._dtype)

        elif callable(value):
            if not self._locked:
                raise RuntimeError('')

        elif value is None:
            pass

        else:
            raise RuntimeError('{}'.format(value))

        return value

    @value.setter
    def value(self, value):
        if self._locked:
            raise RuntimeError('')
        self._value = self._validate(value)

    def copy(self):
        return self.__class__(value=self._value, dtype=self._dtype, locked=self._locked, dimensions=self._dimensions)


def xy_property(component, name):
    def getter(self):
        return getattr(self, name)[component]

    def setter(self, value):
        new = getattr(self, name).copy()
        new[component] = value
        setattr(self, name, new)

    return property(getter, setter)


class Grid(Observable):

    def __init__(self, extent=None, gpts=None, sampling=None, dimensions=2):
        Observable.__init__(self)

        self._dimensions = dimensions

        if isinstance(extent, GridProperty):
            self._extent = extent
        else:
            self._extent = GridProperty(extent, np.float32, locked=False, dimensions=dimensions)

        if isinstance(gpts, GridProperty):
            self._gpts = gpts
        else:
            self._gpts = GridProperty(gpts, np.int32, locked=False, dimensions=dimensions)

        if isinstance(sampling, GridProperty):
            self._sampling = sampling
        else:
            self._sampling = GridProperty(sampling, np.float32, locked=False, dimensions=dimensions)

        if self.extent is None:
            if not ((self.gpts is None) | (self.sampling is None)):
                self._extent.value = self._adjusted_extent()

        if self.gpts is None:
            if not ((self.extent is None) | (self.sampling is None)):
                self._gpts.value = self._adjusted_gpts()

        if self.sampling is None:
            if not ((self.extent is None) | (self.gpts is None)):
                self._sampling.value = self._adjusted_sampling()

        if (extent is not None) & (self.gpts is not None):
            self._sampling.value = self._adjusted_sampling()

        if (gpts is not None) & (self.extent is not None):
            self._sampling.value = self._adjusted_sampling()

    @property
    def extent(self):
        if self._gpts.locked & self._sampling.locked:
            return self._adjusted_extent()

        return self._extent.value

    @extent.setter
    def extent(self, value):
        old = self._extent.value
        self._extent.value = value

        if self._gpts.locked & self._sampling.locked:
            raise RuntimeError()

        if not (self._gpts.locked | (self.extent is None) | (self.sampling is None)):
            self._gpts.value = self._adjusted_gpts()
            self._sampling.value = self._adjusted_sampling()

        elif not (self._sampling.locked | (self.extent is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling()

        self.notify_observers({'name': 'extent', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def gpts(self):
        if self._extent.locked & self._sampling.locked:
            return self._adjusted_sampling()

        return self._gpts.value

    @gpts.setter
    def gpts(self, value):
        old = self._gpts.value
        self._gpts.value = value

        if self._extent.locked & self._sampling.locked:
            raise RuntimeError()

        if not (self._sampling.locked | (self.extent is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling()

        elif not (self._extent.locked | (self.gpts is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent()

        self.notify_observers({'name': 'gpts', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def sampling(self):
        if self._gpts.locked & self._extent.locked:
            return self._adjusted_sampling()

        return self._sampling.value

    @sampling.setter
    def sampling(self, value):
        old = self._sampling.value
        self._sampling.value = value

        if self._gpts.locked & self._extent.locked:
            raise RuntimeError()

        if not (self._gpts.locked | (self.extent is None) | (self.sampling is None)):
            self._gpts.value = self._adjusted_gpts()
            self._extent.value = self._adjusted_extent()

        elif not (self._extent.locked | (self.gpts is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent()

        self.notify_observers({'name': 'sampling', 'old': old, 'new': value, 'change': np.any(old != value)})

    def _adjusted_extent(self):
        return np.float32(self.gpts) * self.sampling

    def _adjusted_gpts(self):
        return np.ceil(self.extent / self.sampling).astype(np.int32)

    def _adjusted_sampling(self):
        return self.extent / np.float32(self.gpts)

    def linspace(self):
        return tuple([linspace_no_endpoint(0., self.extent[i], self.gpts[i]) for i in range(self._dimensions)])

    def fftfreq(self):
        return fftfreq(self.gpts[0], self.sampling[0]), fftfreq(self.gpts[1], self.sampling[1])

    def fftfreq_range(self):
        return np.hstack((fftfreq_range(self.extent[0], self.gpts[0]),
                          fftfreq_range(self.extent[1], self.gpts[1])))

    def check_is_defined(self):
        if (self.extent is None) | (self.gpts is None) | (self.sampling is None):
            raise RuntimeError('grid is not defined')

    def clear(self):
        self._extent.value = None
        self._gpts.value = None
        self._sampling.value = None
        self.notify_observers({'change': True})

    def match(self, other):
        if self.extent is None:
            self.extent = other.extent

        elif other.extent is None:
            other.extent = self.extent

        elif np.any(self.extent != other.extent):
            raise RuntimeError('inconsistent grids')

        if self.gpts is None:
            self.gpts = other.gpts

        elif other.gpts is None:
            other.gpts = self.gpts

        elif np.any(self.gpts != other.gpts):
            raise RuntimeError('inconsistent grids')

        if self.sampling is None:
            self.sampling = other.sampling

        elif other.sampling is None:
            other.sampling = self.sampling

        elif np.any(self.sampling != other.sampling):
            raise RuntimeError('inconsistent grids')

    def copy(self):
        return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
                              dimensions=self._dimensions)

    x_extent = xy_property(0, 'extent')
    y_extent = xy_property(1, 'extent')

    x_gpts = xy_property(0, 'gpts')
    y_gpts = xy_property(1, 'gpts')

    x_sampling = xy_property(0, 'sampling')
    y_sampling = xy_property(1, 'sampling')


class HasGrid(object):

    def __init__(self, extent=None, gpts=None, sampling=None, grid=None, dimensions=2):

        if grid is None:
            self._grid = Grid(extent=extent, gpts=gpts, sampling=sampling, dimensions=dimensions)

        else:
            self._grid = grid

    extent = referenced_property('_grid', 'extent')
    gpts = referenced_property('_grid', 'gpts')
    sampling = referenced_property('_grid', 'sampling')

    @property
    def grid(self):
        return self._grid

    def linspace(self):
        return self._grid.linspace()

    def fftfreq(self):
        return self._grid.fftfreq()

    def check_is_defined(self):
        self._grid.check_is_defined()


def energy2mass(energy):
    """
    Calculate relativistic mass from energy.

    Parameters
    ----------
    energy : scalar
        Energy in electron volt.

    Returns
    -------
    mass : scalar
        Relativistic mass in kg.

    """
    return (1 + units._e * energy / (units._me * units._c ** 2)) * units._me


def energy2wavelength(energy):
    """
    Calculate relativistic de Broglie wavelength from energy.

    Parameters
    ----------
    energy : scalar
        Energy in electron volt.

    Returns
    -------
    wavelength : scalar
        Relativistic de Broglie wavelength in Angstrom.

    """
    return units._hplanck * units._c / np.sqrt(
        energy * (2 * units._me * units._c ** 2 / units._e + energy)) / units._e * 1.e10


def energy2sigma(energy):
    """
    Calculate interaction parameter from energy.

    Parameters
    ----------
    energy : scalar
        Energy in electron volt.

    Returns
    -------
    interaction parameter : scalar
        Interaction parameter in 1 / (Angstrom * eV).

    """
    return (2 * np.pi * energy2mass(energy) * units.kg * units._e * units.C * energy2wavelength(energy) / (
            units._hplanck * units.s * units.J) ** 2)


class EnergyWrapper(Observable):

    def __init__(self, energy=None):
        Observable.__init__(self)

        self._energy = energy

    energy = notifying_property('_energy')

    @property
    def wavelength(self):
        self.check_is_defined()
        return energy2wavelength(self.energy)

    @property
    def sigma(self):
        self.check_is_defined()
        return energy2sigma(self.energy)

    def check_is_defined(self):
        if self.energy is None:
            raise RuntimeError('energy is not defined')

    def match(self, other):
        if other.energy is None:
            other.energy = self.energy

        elif self.energy is None:
            self.energy = other.energy

        elif self.energy != other.energy:
            raise RuntimeError('inconsistent energies')

    def copy(self):
        return self.__class__(self.energy)


class HasEnergy(object):

    def __init__(self, energy=None, energy_wrapper=None):
        if energy_wrapper is None:
            self._energy_wrapper = EnergyWrapper(energy=energy)

        else:
            self._energy_wrapper = energy_wrapper

    energy = referenced_property('_energy_wrapper', 'energy')
    wavelength = referenced_property('_energy_wrapper', 'wavelength')
    sigma = referenced_property('_energy_wrapper', 'sigma')

    @property
    def energy_wrapper(self):
        return self._energy_wrapper

    def check_is_defined(self):
        self._energy_wrapper.check_is_defined()


class HasGridAndEnergy(HasGrid, HasEnergy):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, grid=None, energy_wrapper=None):
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid)
        HasEnergy.__init__(self, energy=energy, energy_wrapper=energy_wrapper)

    def check_is_defined(self):
        self.grid.check_is_defined()
        self.energy_wrapper.check_is_defined()

    def match_grid_and_energy(self, other):
        try:
            self.grid.match(other.grid)
        except AttributeError:
            pass

        try:
            self.energy_wrapper.match(other.energy_wrapper)
        except AttributeError:
            pass

    def semiangles(self):
        kx, ky = self.grid.fftfreq()
        wavelength = self.wavelength
        return kx * wavelength, ky * wavelength


class TensorBase(HasGrid):
    def __init__(self, tensor, tensor_dimensions, spatial_dimensions, extent=None, sampling=None, space='direct'):
        if len(tensor.shape) != tensor_dimensions:
            raise RuntimeError('tensor shape {} not {}d'.format(tensor.shape, tensor_dimensions))
        self._tensor = tensor

        gpts = GridProperty(lambda: self.gpts, dtype=np.int32, locked=True, dimensions=spatial_dimensions)

        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, dimensions=spatial_dimensions)

        self.space = space

    @property
    def gpts(self):
        raise NotImplementedError()

    def tensor(self):
        return self._tensor

    def numpy(self):
        return self.tensor().numpy()

    def intensity(self):
        new_tensor = self.__class__(tensor=tf.abs(self.tensor()) ** 2, space=self.space)
        new_tensor._grid = self._grid.copy()
        return new_tensor

    def copy(self):
        new_tensor = self.__class__(tensor=tf.identity(self.tensor()), extent=self.extent.copy(), space=self.space)
        new_tensor._grid = self._grid.copy()
        return new_tensor

    def show(self, *args):
        raise NotImplementedError()


class TensorWithGrid(TensorBase):

    def __init__(self, tensor, extent=None, sampling=None, space='direct'):
        TensorBase.__init__(self, tensor=tensor, tensor_dimensions=3, spatial_dimensions=2, extent=extent,
                            sampling=sampling, space=space)

    @property
    def gpts(self):
        return np.array([dim for dim in self.tensor().shape[1:]])

    def show(self, fig_scale=1, space='direct', scale='linear', mode='intensity', tile=(1, 1)):
        return show_array(self.numpy(), self.extent, self.space, fig_scale=fig_scale, display_space=space,
                          mode=mode, scale=scale, tile=tile)

    def profile(self, direction='x', displacement=None):

        if displacement:
            raise NotImplementedError()

        if direction is 'x':
            tensor = self._tensor[0, :, self._tensor.shape[2] // 2]
            extent = self.extent[0]

        elif direction is 'y':
            tensor = self._tensor[0, self._tensor.shape[1] // 2, :]
            extent = self.extent[1]

        else:
            raise RuntimeError('direction {} nor recognized'.format(direction))

        return LineProfile(tensor=tensor, extent=np.array([extent]))


class TensorWithGridAndEnergy(TensorWithGrid, HasGridAndEnergy):

    def __init__(self, tensor, extent=None, sampling=None, energy=None, space='direct'):
        HasGridAndEnergy.__init__(self, energy=energy, extent=extent, sampling=sampling)
        TensorWithGrid.__init__(self, tensor=tensor, extent=extent, sampling=sampling, space=space)


class LineProfile(TensorBase):

    def __init__(self, tensor, extent=None, sampling=None, space='direct'):
        tensor = tf.cast(tensor, tf.float32)
        TensorBase.__init__(self, tensor=tensor, tensor_dimensions=1, spatial_dimensions=1, extent=extent,
                            sampling=sampling, space=space)

    def __len__(self):
        return len(self._tensor)

    @property
    def gpts(self):
        return np.array([self.tensor().shape[0]])

    def show(self, mode='mag', *args, **kwargs):
        y = self.numpy()
        x = np.linspace(0., self.extent[0], self.gpts[0])
        show_line(x, y, mode, *args, **kwargs)


class Interactive(object):

    def get_observables(self):
        raise NotImplementedError()


class FrequencyTransfer(TensorFactory, Observable):

    def __init__(self, save_tensor=True):
        TensorFactory.__init__(self, save_tensor=save_tensor)
        Observable.__init__(self)

        self.observe(self)


class PrismCoefficients(TensorFactory, Observable):

    def __init__(self, kx, ky, save_tensor=True):
        self._kx = kx
        self._ky = ky

        TensorFactory.__init__(self, save_tensor=save_tensor)
        Observable.__init__(self)

        self.register_observer(self)

    def check_is_defined(self):
        if (self.kx is None) | (self.ky is None):
            raise RuntimeError('')

    @property
    def kx(self):
        return self._kx

    @property
    def ky(self):
        return self._ky
