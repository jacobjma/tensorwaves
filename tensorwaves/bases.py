from numbers import Number

import numpy as np
import tensorflow as tf

from tensorwaves.plotutils import show_array
from tensorwaves.utils import energy2wavelength, energy2sigma, linspace_no_endpoint, fftfreq, freq2angles, \
    fourier_propagator, complex_exponential


class Observable(object):
    def __init__(self):
        self._observers = []

    def register_observer(self, observer):
        self._observers.append(observer)

    def notify_observers(self, message, exclude_self=False):
        for observer in self._observers:
            if not ((observer is self) & exclude_self):
                observer.notify(self, message)


class Observer(object):
    def __init__(self, observable=None):
        if observable:
            observable.register_observer(self)

    def notify(self, observable, message):
        raise NotImplementedError()


def base_property(name):
    def getter(self):
        return getattr(self, name)

    def setter(self, value):
        setattr(self, name, value)

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


class Updatable(Observer, Observable):

    def __init__(self):
        Observable.__init__(self)
        Observer.__init__(self)

        self._up_to_date = False
        self._forward = True

    @property
    def up_to_date(self):
        return self._up_to_date

    @up_to_date.setter
    def up_to_date(self, value):
        self._up_to_date = value

    def notify(self, observable, message):
        if message['change']:
            self.up_to_date = False
            self.notify_observers(message, exclude_self=True)

            # if self._mediator:
            #    self._mediator.notify(self, message)


class HasData(Updatable):

    def __init__(self, save_data=True):

        Updatable.__init__(self)

        self._save_data = save_data
        self._data = None

    def clear_data(self):
        self._data = None

    def _calculate_data(self):
        return None

    def get_data(self):
        if (self.up_to_date & self._save_data):
            data = self._data
        else:
            data = self._calculate_data()
            if self._save_data:
                self._data = data
                self.up_to_date = True
        return data


def xy_property(component, name):
    def getter(self):
        return getattr(self, name)[component]

    def setter(self, value):
        new = getattr(self, name).copy()
        new[component] = value
        setattr(self, name, new)

    return property(getter, setter)


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
                raise RuntimeError()
            value = np.array(value).astype(self._dtype)

        elif isinstance(value, Number):
            value = np.full(2, value, dtype=self._dtype)

        elif (value is None) | callable(value):
            pass

        else:
            raise RuntimeError()

        return value

    @value.setter
    def value(self, value):
        if self._locked:
            raise RuntimeError()
        self._value = self._validate(value)

    def copy(self):
        return self.__class__(value=self._value, dtype=self._dtype, locked=self._locked)


class GridException(Exception):
    pass


class Grid(Observable):

    def __init__(self, extent=None, gpts=None, sampling=None, space='direct'):
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

        self._space = space

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

        elif not (self._extent.locked | (self.gpts is None) | (self.sampling is None)):
            self._extent.value = self._adjusted_extent()

        self.notify_observers({'name': 'sampling', 'old': old, 'new': value, 'change': np.any(old != value)})

    @property
    def space(self):
        return self._space

    def check_is_defined(self):
        if ((self.extent is None) | (self.gpts is None) | (self.sampling is None)):
            raise GridException()  # ('grid is not defined')

    def _adjusted_extent(self):
        return np.float32(self.gpts) * self.sampling

    def _adjusted_gpts(self):
        return np.ceil(self.extent / self.sampling).astype(np.int32)

    def _adjusted_sampling(self):
        return self.extent / np.float32(self.gpts)

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

    def copy(self):
        return self.__class__(extent=self._extent.copy(), gpts=self._gpts.copy(), sampling=self._sampling.copy(),
                              space=self._space)

    def is_compatible(self, other):
        pass

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

    x_extent = xy_property(0, 'extent')
    y_extent = xy_property(1, 'extent')

    x_gpts = xy_property(0, 'gpts')
    y_gpts = xy_property(1, 'gpts')

    x_sampling = xy_property(0, 'sampling')
    y_sampling = xy_property(1, 'sampling')


class HasGrid(object):

    def __init__(self, extent=None, gpts=None, sampling=None, space=None, grid=None):
        if grid is None:
            grid = Grid(extent=extent, gpts=gpts, sampling=sampling, space=space)

        self._grid = grid

    @property
    def grid(self):
        return self._grid


class Accelerator(Observable):

    def __init__(self, energy=None):
        Observable.__init__(self)

        self._energy = energy

    energy = notifying_property('_energy')

    def check_is_defined(self):
        if self.energy is None:
            raise RuntimeError('energy is not defined')

    @property
    def wavelength(self):
        return energy2wavelength(self.energy)

    @property
    def interaction_parameter(self):
        return energy2sigma(self.energy)

    def match(self, other):
        if self.energy is None:
            self.energy = other.energy
        elif other.energy is None:
            other.energy = self.energy

    def copy(self):
        return self.__class__(self.energy)


class HasAccelerator(object):

    def __init__(self, energy=None, accelerator=None):
        if accelerator is None:
            accelerator = Accelerator(energy=energy)

        self._accelerator = accelerator

    @property
    def accelerator(self):
        return self._accelerator


class Tensor(HasGrid):

    def __init__(self, tensor, extent=None, sampling=None, space=None, grid=None):
        self._tensor = tensor

        gpts = GridProperty(lambda: np.array([dim.value for dim in tensor.shape[1:]]), dtype=np.int32, locked=True)

        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, space=space, grid=grid)
        # Observable.__init__(self)

    def check_is_defined(self):
        self._grid.check_is_defined()

    def tensorflow(self):
        return self._tensor

    def numpy(self):
        return self._tensor.numpy()

    def show(self, i=None, space='direct', mode='magnitude', **kwargs):
        array = self.numpy()
        if i is not None:
            array = array[i][None]

        show_array(array, extent=self.grid.extent, space=self.grid.space, display_space=space, mode=mode, **kwargs)

    def copy(self):
        return self.__class__(tensor=tf.identity(self._tensor), grid=self.grid.copy())


class TensorWaves(Tensor, HasAccelerator):

    def __init__(self, tensor, extent=None, sampling=None, energy=None, grid=None, accelerator=None):
        Tensor.__init__(self, tensor, extent=extent, sampling=sampling, space='direct', grid=grid)

        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

    def get_tensor(self):
        return self

    def multislice(self, potential, in_place=False):

        self.grid.match(potential.grid)

        if in_place:
            wave = self
        else:
            wave = self.copy()

        for potential_slice in potential.slice_generator():
            wave._tensor = wave._tensor * complex_exponential(wave.accelerator.interaction_parameter *
                                                              potential_slice)
            wave.propagate(potential.slice_thickness)

        return wave

    def propagate(self, dz):
        self._tensor = self._fourier_convolution(self.fourier_propagator(dz))

    def fourier_propagator(self, dz):
        kx, ky = self.grid.fftfreq()
        return fourier_propagator(((kx ** 2)[:, None] + (ky ** 2)[None, :]), dz,
                                  self.accelerator.wavelength)[None, :, :]

    def _fourier_convolution(self, propagator):
        return tf.ifft2d(tf.fft2d(self._tensor) * propagator)

    def _get_show_data(self):
        return self.get_tensor().numpy()

    #     #
    #     # def apply_ctf(self, ctf=None, in_place=False, aperture_radius=None, aperture_rolloff=0., **kwargs):
    #     #     if ctf is None:
    #     #         ctf = CTF(extent=self.extent, gpts=self.gpts, energy=self.energy, aperture_radius=aperture_radius,
    #     #                   aperture_rolloff=aperture_rolloff, **kwargs)
    #     #     else:
    #     #         ctf.adopt_grid(self)
    #     #         ctf.energy = self.energy
    #     #
    #     #     return self.convolve(ctf.get_tensor(), in_place=in_place)
    #
    #     # def detect(self):
    #     #     return Image(tf.abs(self._tensor) ** 2, extent=self._extent.copy(), sampling=self._sampling.copy())
    #
    def copy(self):
        return self.__class__(tensor=tf.identity(self._tensor), grid=self.grid.copy(),
                              accelerator=self.accelerator.copy())


class FrequencyMultiplier(HasData, HasGrid, HasAccelerator):

    def __init__(self, extent=None, gpts=None, sampling=None, energy=None, save_data=True, grid=None, accelerator=None):

        HasData.__init__(self, save_data=save_data)
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, space='fourier', grid=grid)
        HasAccelerator.__init__(self, energy=energy, accelerator=accelerator)

        self.grid.register_observer(self)
        self.accelerator.register_observer(self)
        self.register_observer(self)

    def get_semiangles(self, return_squared_norm=False, return_azimuth=False):
        return freq2angles(*self.grid.fftfreq(), self.accelerator.wavelength, return_squared_norm, return_azimuth)

    def apply(self, wave, in_place=False):
        wave.grid.match(self.grid)
        #self.grid.match(wave.grid)
        #print(self.grid)
        self.accelerator.match(wave.accelerator)

        wave = wave.get_tensor()
        tensor = tf.ifft2d(tf.fft2d(wave._tensor) * self.get_data())
        if in_place:
            wave._tensor = tensor
        else:
            wave = TensorWaves(tensor, extent=self.grid.extent, energy=self.accelerator.energy)
        return wave

    def show(self, space='direct', mode='magnitude', **kwargs):
        self._show_data(space=space, mode=mode, **kwargs)

    def _get_show_data(self):
        return self.get_data().numpy()

    def _show_data(self, space, mode, **kwargs):
        show_array(array=self._get_show_data(), extent=self.grid.extent, space=self.grid.space, display_space=space,
                   mode=mode, **kwargs)
