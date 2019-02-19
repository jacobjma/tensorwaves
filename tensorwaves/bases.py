import numpy as np
import tensorflow as tf
from ase import units


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
        change = old != value
        self.notify_observers({'name': name, 'old': old, 'new': value, 'change': change})

    return property(getter, setter)


class Observable(object):
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, message):
        for observer in self._observers:
            observer.notify(self, message)


class Observer(object):
    def __init__(self, observable=None):
        if observable:
            observable.register_observer(self)

    def notify(self, observable, message):
        raise NotImplementedError()


class TensorFactory(Observer):

    def __init__(self, save_tensor=True):
        Observer.__init__(self)
        self.up_to_date = False
        self._save_tensor = save_tensor
        self._tensor = None

    def notify(self, observable, message):
        if message['change']:
            self.up_to_date = False

    def _calculate_tensor(self):
        raise NotImplementedError()

    def get_tensor(self):
        if self.up_to_date & self._save_tensor:
            data = self._tensor
        else:
            data = self._calculate_tensor()
            if self._save_tensor:
                self._tensor = data
                self.up_to_date = True
        return data

    def clear_tensor(self):
        self._tensor = None
        self.up_to_date = False

    def tensorflow(self):
        return self.get_tensor().tensorflow()

    def numpy(self):
        return self.get_tensor().numpy()


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
    return tf.lin_space(start, interval - interval / tf.cast(num, tf.float32), num)


def fftfreq(n, d):
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
    p1 = tf.lin_space(0., m - 1, m)
    p2 = tf.lin_space(-tf.cast(n // 2, tf.float32), -1, n // 2)
    return tf.concat((p1, p2), axis=0) / (n * d)


def fftfreq_range(extent, n):
    return np.array([-n / extent, n / extent]) / 2.


class GridProperty(object):

    def __init__(self, value, dtype, locked=False):
        self._dtype = dtype
        self._locked = locked
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
            if len(value) != 2:
                raise RuntimeError('')
            value = np.array(value).astype(self._dtype)

        elif isinstance(value, (int, float, complex)):
            value = np.full(2, value, dtype=self._dtype)

        elif callable(value):
            if not self._locked:
                raise RuntimeError('')

        elif value is None:
            pass

        else:
            raise RuntimeError('')

        return value

    @value.setter
    def value(self, value):
        if self._locked:
            raise RuntimeError('')
        self._value = self._validate(value)

    def copy(self):
        return self.__class__(value=self._value, dtype=self._dtype, locked=self._locked)


def xy_property(component, name):
    def getter(self):
        return getattr(self, name)[component]

    def setter(self, value):
        new = getattr(self, name).copy()
        new[component] = value
        setattr(self, name, new)

    return property(getter, setter)


class Grid(Observable):

    def __init__(self, extent=None, gpts=None, sampling=None):
        Observable.__init__(self)

        if isinstance(extent, GridProperty):
            self._extent = extent
        else:
            self._extent = GridProperty(extent, np.float32, locked=False)

        if isinstance(gpts, GridProperty):
            self._gpts = gpts
        else:
            self._gpts = GridProperty(gpts, np.int32, locked=False)

        if isinstance(sampling, GridProperty):
            self._sampling = sampling
        else:
            self._sampling = GridProperty(sampling, np.float32, locked=False)

        if self.extent is None:
            if not ((self.gpts is None) | (self.sampling is None)):
                self._extent.value = self._adjusted_extent()

        if self.gpts is None:
            if not ((self.extent is None) | (self.sampling is None)):
                self._gpts.value = self._adjusted_gpts()

        if self.sampling is None:
            if not ((self.extent is None) | (self.gpts is None)):
                self._sampling.value = self._adjusted_sampling()

    @property
    def extent(self):
        return self._extent.value

    @extent.setter
    def extent(self, value):
        old = self._extent.value
        self._extent.value = value

        if not (self._gpts.locked | (self.extent is None) | (self.sampling is None)):
            self._gpts.value = self._adjusted_gpts()
            self._sampling.value = self._adjusted_sampling()

        elif not (self._sampling.locked | (self.extent is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling()

        self.notify_observers({'name': 'extent', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def gpts(self):
        return self._gpts.value

    @gpts.setter
    def gpts(self, value):
        old = self._gpts.value
        self._gpts.value = value

        if not (self._sampling.locked | (self.extent is None) | (self.gpts is None)):
            self._sampling.value = self._adjusted_sampling()

        elif not (self._extent.locked | (self.gpts is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent()

        self.notify_observers({'name': 'gpts', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def sampling(self):
        return self._sampling.value

    @sampling.setter
    def sampling(self, value):
        old = self._sampling.value
        self._sampling.value = value

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
        return (linspace_no_endpoint(0., self.extent[0], self.gpts[0]),
                linspace_no_endpoint(0., self.extent[1], self.gpts[1]))

    def fftfreq(self):
        return fftfreq(self.gpts[0], self.sampling[0]), fftfreq(self.gpts[1], self.sampling[1])

    def fftfreq_range(self):
        return np.hstack((fftfreq_range(self.extent[0], self.gpts[0]),
                          fftfreq_range(self.extent[1], self.gpts[1])))

    def check_is_defined(self):
        if (self.extent is None) | (self.gpts is None) | (self.sampling is None):
            raise RuntimeError('grid is not defined')

    def match(self, other):
        if (self.extent is None) & (not self._extent.locked):
            self.extent = other.extent
        elif not other._extent.locked:
            other.extent = self.extent

        if (self.gpts is None) & (not self._gpts.locked):
            self.gpts = other.gpts
        elif not other._gpts.locked:
            other.gpts = self.gpts

        if (self.sampling is None) & (not self._sampling.locked):
            self.sampling = other.sampling
        elif not other._sampling.locked:
            other.sampling = self.sampling

    def copy(self):
        return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy())

    x_extent = xy_property(0, 'extent')
    y_extent = xy_property(1, 'extent')

    x_gpts = xy_property(0, 'gpts')
    y_gpts = xy_property(1, 'gpts')

    x_sampling = xy_property(0, 'sampling')
    y_sampling = xy_property(1, 'sampling')


class HasGrid(object):

    def __init__(self, extent=None, gpts=None, sampling=None, grid=None):
        if grid is None:
            grid = Grid(extent=extent, gpts=gpts, sampling=sampling)

        self._grid = grid

    extent = referenced_property('_grid', 'extent')
    gpts = referenced_property('_grid', 'gpts')
    sampling = referenced_property('_grid', 'sampling')

    def linspace(self):
        return self._grid.linspace()

    def fftfreq(self):
        return self._grid.fftfreq()


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


class EnergyProperty(Observable):

    def __init__(self, energy=None):
        Observable.__init__(self)

        self._energy = energy

    energy = notifying_property('_energy')

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    @property
    def interaction_parameter(self):
        return energy2sigma(self.energy)

    def check_is_defined(self):
        if self.energy is None:
            raise RuntimeError('energy is not defined')

    def match(self, other):
        if self.energy is None:
            self.energy = other.energy
        elif other.energy is None:
            other.energy = self.energy

    def copy(self):
        return self.__class__(self.energy)


class HasEnergy(object):

    def __init__(self, energy=None, energy_property=None):
        if energy_property is None:
            energy_property = EnergyProperty(energy=energy)

        elif not isinstance(energy_property, EnergyProperty):
            raise RuntimeError('')

        self._energy = energy_property

    energy = referenced_property('_energy', 'energy')
    wavelength = referenced_property('_energy', 'wavelength')
    sigma = referenced_property('_energy', 'sigma')


class Tensor(HasGrid):

    def __init__(self, tensor, extent=None, sampling=None, space='direct'):
        self._tensor = tensor

        gpts = GridProperty(lambda: self.gpts, dtype=np.int32, locked=True)

        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)

        self.space = space

    @property
    def gpts(self):
        return np.array([dim.value for dim in self._tensor.shape[1:]])

    def check_is_defined(self):
        self._grid.check_is_defined()

    def tensorflow(self):
        return self._tensor

    def numpy(self):
        return self._tensor.numpy()

    def copy(self):
        new_tensor = self.__class__(tensor=tf.identity(self._tensor), extent=self.extent.copy(), space=self.space)
        new_tensor._grid = self._grid.copy()
        return new_tensor

    def show(self, i):
        pass


class TensorWithEnergy(Tensor, HasEnergy):

    def __init__(self, tensor, extent=None, sampling=None, energy=None, space='direct'):
        Tensor.__init__(self, tensor=tensor, extent=extent, sampling=sampling, space=space)
        HasEnergy.__init__(self, energy=energy)

    def check_is_defined(self):
        self._grid.check_is_defined()
        self._energy.check_is_defined()

# class Showable(HasGrid):
#
#     def __init__(self, extent=None, gpts=None, sampling=None, grid=None, space=None):
#         self._space = space
#
#         HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid)
#
#     @property
#     def space(self):
#         return self._space
#
#     def get_showable_tensor(self, i=0):
#         raise NotImplementedError()
#
#     def show(self, i=None, space='direct', mode='magnitude', scale='linear', fig_scale=1, **kwargs):
#         from tensorwaves.plotutils import show_array
#         array = self.get_showable_tensor(i).numpy()
#
#         # if i is not None:
#         #    array = array[i][None]
#
#         show_array(array, extent=self.grid.extent, space=self.space, display_space=space, mode=mode, scale=scale,
#                    fig_scale=fig_scale, **kwargs)
#
#
# class ShowableWithEnergy(Showable, HasAccelerator):
#
#     def __init__(self, extent=None, gpts=None, sampling=None, grid=None, space=None, energy=None, accelerator=None):
#         Showable.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid, space=space)
#         HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)
#
#     def match(self, other):
#         other.grid.match(self.grid)
#         other.accelerator.match(self.accelerator)
#
#

#
#
# class TensorWithEnergy(Tensor, HasAccelerator):
#
#     def __init__(self, tensor, extent=None, sampling=None, grid=None, space=None, energy=None, accelerator=None):
#         Tensor.__init__(self, tensor, extent=extent, sampling=sampling, space=space, grid=grid)
#         HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)
#
#
#
# class FrequencyMultiplier(HasData, Showable, HasAccelerator):
#
#     def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_data=True, grid=None, accelerator=None):
#
#         HasData.__init__(self, save_data=save_data)
#         Showable.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid, space='fourier')
#         HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)
#
#         self.grid.register_observer(self)
#         self.accelerator.register_observer(self)
#
#         self._observing = [self.grid, self.accelerator]
#
#         self.register_observer(self)
#
#     def get_semiangles(self, return_squared_norm=False, return_azimuth=False):
#         kx, ky = self._grid.fftfreq()
#
#         return freq2angles(kx=kx, ky=ky, wavelength=self._accelerator.wavelength,
#                            return_squared_norm=return_squared_norm, return_azimuth=return_azimuth)
#
#     def apply(self, wave, in_place=False):
#         wave.grid.match(self.grid)
#         wave.accelerator.match(self.accelerator)
#
#         wave = wave.get_tensor()
#
#         tensor = tf.ifft2d(tf.fft2d(wave._tensor) * self.get_data()._tensor)
#
#         if in_place:
#             wave._tensor = tensor
#         else:
#             wave = TensorWaves(tensor, extent=self.grid.extent.copy(), energy=self.accelerator.energy)
#
#         return wave
#
#     def get_showable_tensor(self):
#         return self.get_data()
