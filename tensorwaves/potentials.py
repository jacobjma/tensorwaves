import csv
import os

import numpy as np
import tensorflow as tf
from ase import units
from scipy.optimize import brentq

from tensorwaves.bases import Tensor, TensorFactory, HasGrid, GridProperty
from tensorwaves.interpolate_spline import interpolate_spline
from tensorwaves.utils import batch_generator

eps0 = units._eps0 * units.A ** 2 * units.s ** 4 / (units.kg * units.m ** 3)
kappa = 4 * np.pi * eps0 / (2 * np.pi * units.Bohr * units._e * units.C)


def log_grid(start, stop, num):
    """
    Return numbers spaced evenly on a log scale over the closed interval [start, stop].

    Parameters
    ----------
    start : scalar
        The starting value of the interval.
    stop : scalar
        The end value of the interval.
    num : int
        Number of samples to generate.

    Returns
    -------
    samples : tensor
        There are num samples in the closed interval [start, stop] spaced evenly on a logarithmic scale.

    """
    start = tf.cast(start, tf.float32)
    stop = tf.cast(stop, tf.float32)
    dt = tf.math.log(stop / start) / (num - 1)
    return tf.cast(start * tf.exp(dt * tf.linspace(0., num - 1, num)), tf.float32)


class PotentialParameterization(object):

    def __init__(self, filename=None, parameters=None, tolerance=1e-2):

        self._tolerance = tolerance
        self._cutoffs = {}
        self._functions = {}
        self._soft_functions = {}

        self._filename = filename
        if parameters is None:
            self.load_parameters()
        else:
            self._parameters = parameters

    @property
    def parameters(self):
        return self._parameters

    @property
    def tolerance(self):
        return self._tolerance

    def load_parameters(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self._filename)
        parameters = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            keys = next(reader)
            for _, row in enumerate(reader):
                values = list(map(float, row))
                parameters[int(row[0])] = dict(zip(keys, values))
        self._parameters = parameters

    def _create_projected_function(self, atomic_number):
        raise RuntimeError()

    def _create_function(self, atomic_number):
        raise RuntimeError()

    def _create_soft_function(self, atomic_number, r_cut):
        raise RuntimeError()

    def _find_cutoff(self, atomic_number):
        # print(self.get_function(atomic_number))
        return np.float32(brentq(lambda x: (self.get_function(atomic_number)(x)) - self.tolerance, 1e-7, 1000))

    def get_cutoff(self, atomic_number):
        try:
            return self._cutoffs[atomic_number]
        except KeyError:
            self._cutoffs[atomic_number] = self._find_cutoff(atomic_number)
            return self._cutoffs[atomic_number]

    def get_function(self, atomic_number):
        try:
            return self._functions[atomic_number]
        except KeyError:
            self._functions[atomic_number] = self._create_function(atomic_number)
            return self._functions[atomic_number]

    def get_soft_function(self, atomic_number):
        try:
            return self._soft_functions[atomic_number]
        except KeyError:
            self._soft_functions[atomic_number] = self._create_soft_function(atomic_number,
                                                                             self.get_cutoff(atomic_number))
            return self._soft_functions[atomic_number]


class LobatoPotential(PotentialParameterization):

    def __init__(self, tolerance=1e-3):
        PotentialParameterization.__init__(self, filename='data/lobato.txt', tolerance=tolerance)

    def _convert_coefficients(self, atomic_number):
        a = [np.pi ** 2 * self.parameters[atomic_number][key_a] / self.parameters[atomic_number][key_b] ** (3 / 2.)
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]

        b = [2 * np.pi / tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

        return (tf.cast(x, dtype=tf.float32) for x in (a, b))

    def _create_function(self, atomic_number):
        a, b = self._convert_coefficients(atomic_number)

        def func(r):
            return (a[0] * (2. / (b[0] * r) + 1.) * tf.exp(-b[0] * r) +
                    a[1] * (2. / (b[1] * r) + 1.) * tf.exp(-b[1] * r) +
                    a[2] * (2. / (b[2] * r) + 1.) * tf.exp(-b[2] * r) +
                    a[3] * (2. / (b[3] * r) + 1.) * tf.exp(-b[3] * r) +
                    a[4] * (2. / (b[4] * r) + 1.) * tf.exp(-b[4] * r))

        return func

    def _create_soft_function(self, atomic_number, r_cut):
        a, b = self._convert_coefficients(atomic_number)

        r_cut = tf.cast(r_cut, tf.float32)

        dvdr_cut = - (a[0] * (2. / (b[0] * r_cut ** 2) + 2. / r_cut + b[0]) * tf.exp(-b[0] * r_cut) +
                      a[1] * (2. / (b[1] * r_cut ** 2) + 2. / r_cut + b[1]) * tf.exp(-b[1] * r_cut) +
                      a[2] * (2. / (b[2] * r_cut ** 2) + 2. / r_cut + b[2]) * tf.exp(-b[2] * r_cut) +
                      a[3] * (2. / (b[3] * r_cut ** 2) + 2. / r_cut + b[3]) * tf.exp(-b[3] * r_cut) +
                      a[4] * (2. / (b[4] * r_cut ** 2) + 2. / r_cut + b[4]) * tf.exp(-b[4] * r_cut))

        v_cut = (a[0] * (2. / (b[0] * r_cut) + 1.) * tf.exp(-b[0] * r_cut) +
                 a[1] * (2. / (b[1] * r_cut) + 1.) * tf.exp(-b[1] * r_cut) +
                 a[2] * (2. / (b[2] * r_cut) + 1.) * tf.exp(-b[2] * r_cut) +
                 a[3] * (2. / (b[3] * r_cut) + 1.) * tf.exp(-b[3] * r_cut) +
                 a[4] * (2. / (b[4] * r_cut) + 1.) * tf.exp(-b[4] * r_cut))

        def func(r):
            r = tf.clip_by_value(r, 0, r_cut)

            v = (a[0] * (2. / (b[0] * r) + 1.) * tf.exp(-b[0] * r) +
                 a[1] * (2. / (b[1] * r) + 1.) * tf.exp(-b[1] * r) +
                 a[2] * (2. / (b[2] * r) + 1.) * tf.exp(-b[2] * r) +
                 a[3] * (2. / (b[3] * r) + 1.) * tf.exp(-b[3] * r) +
                 a[4] * (2. / (b[4] * r) + 1.) * tf.exp(-b[4] * r))

            return v - v_cut - (r - r_cut) * dvdr_cut

        return func

    def _create_projected_function(self, atomic_number):
        pass


class KirklandPotential(PotentialParameterization):

    def __init__(self, tolerance=1e-3):
        PotentialParameterization.__init__(self, filename='data/kirkland.txt', tolerance=tolerance)

    def _convert_coefficients(self, atomic_number):
        a = [np.pi * self.parameters[atomic_number][key] for key in ('a1', 'a2', 'a3')]
        b = [2. * np.pi * tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3')]
        c = [np.pi ** (3. / 2.) * self.parameters[atomic_number][key_c] / self.parameters[atomic_number][key_d] ** (
                3. / 2.) for key_c, key_d in
             zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))]
        d = [np.pi ** 2 / self.parameters[atomic_number][key] for key in ('d1', 'd2', 'd3')]

        return (tf.cast(x, dtype=tf.float32) for x in (a, b, c, d))

    def _create_function(self, atomic_number):
        a, b, c, d = self._convert_coefficients(atomic_number)

        def func(r):
            return (a[0] * tf.exp(-b[0] * r) / r + c[0] * tf.exp(-d[0] * r ** 2.) +
                    a[1] * tf.exp(-b[1] * r) / r + c[1] * tf.exp(-d[1] * r ** 2.) +
                    a[2] * tf.exp(-b[2] * r) / r + c[2] * tf.exp(-d[2] * r ** 2.))

        return func

    def _create_soft_function(self, atomic_number, r_cut):
        a, b, c, d = self._convert_coefficients(atomic_number)

        r_cut = tf.cast(r_cut, tf.float32)

        dvdr_cut = (- a[0] * (1 / r_cut + b[0]) * tf.exp(-b[0] * r_cut) / r_cut -
                    2 * c[0] * d[0] * r_cut * tf.exp(-d[0] * r_cut ** 2)
                    - a[1] * (1 / r_cut + b[1]) * tf.exp(-b[1] * r_cut) / r_cut -
                    2 * c[1] * d[1] * r_cut * tf.exp(-d[1] * r_cut ** 2)
                    - a[2] * (1 / r_cut + b[2]) * tf.exp(-b[2] * r_cut) / r_cut -
                    2 * c[2] * d[2] * r_cut * tf.exp(-d[2] * r_cut ** 2))

        v_cut = (a[0] * tf.exp(-b[0] * r_cut) / r_cut + c[0] * tf.exp(-d[0] * r_cut ** 2.) +
                 a[1] * tf.exp(-b[1] * r_cut) / r_cut + c[1] * tf.exp(-d[1] * r_cut ** 2.) +
                 a[2] * tf.exp(-b[2] * r_cut) / r_cut + c[2] * tf.exp(-d[2] * r_cut ** 2.))

        def func(r):
            r = tf.clip_by_value(r, 0, r_cut)

            v = (a[0] * tf.exp(-b[0] * r) / r + c[0] * tf.exp(-d[0] * r ** 2.) +
                 a[1] * tf.exp(-b[1] * r) / r + c[1] * tf.exp(-d[1] * r ** 2.) +
                 a[2] * tf.exp(-b[2] * r) / r + c[2] * tf.exp(-d[2] * r ** 2.))

            return v - v_cut - (r - r_cut) * dvdr_cut

        return func

    def _create_projected_function(self, atomic_number):
        from scipy.special import kn

        a, b, c, d = self._convert_coefficients(atomic_number)

        def func(r):
            v = (2 * a[0] * kn(0, b[0] * r) + tf.sqrt(np.pi / d[0]) * c[0] * tf.exp(-d[0] * r ** 2.) +
                 2 * a[1] * kn(0, b[1] * r) + tf.sqrt(np.pi / d[1]) * c[1] * tf.exp(-d[1] * r ** 2.) +
                 2 * a[2] * kn(0, b[2] * r) + tf.sqrt(np.pi / d[2]) * c[2] * tf.exp(-d[2] * r ** 2.))

            return v

        return func


class PotentialProjector(object):

    def __init__(self, num_samples=200, quadrature='riemann'):

        self._quadrature = quadrature.lower()

        if self._quadrature == 'riemann':
            self._xk = np.linspace(-1., 1., num_samples + 1).astype(np.float32)
            self._wk = self._xk[1:] - self._xk[:-1]
            self._xk = self._xk[:-1]

        elif self._quadrature == 'tanh-sinh':
            raise NotImplementedError()

        else:
            raise RuntimeError()

    def project(self, function, r, a, b):

        c = tf.reshape(((b - a) / 2.), (-1, 1))
        d = tf.reshape(((b + a) / 2.), (-1, 1))

        xkab = self._xk[None, :] * c + d
        wkab = self._wk[None, :] * c

        rxy = tf.sqrt(r[None, None, :] ** 2. + (xkab ** 2.)[:, :, None])
        rxy = tf.clip_by_value(rxy, 0, r[-1])
        return tf.reduce_sum(function(rxy) * wkab[:, :, None], axis=1)


class Potential(HasGrid, TensorFactory):

    def __init__(self, atoms=None, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5,
                 num_slices=None, parametrization='lobato', periodic=True, method='splines', atoms_per_loop=10,
                 tolerance=1e-2, num_nodes=50, save_tensor=True):

        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling)
        TensorFactory.__init__(self, save_tensor=save_tensor)

        self._atoms = None
        self._species = None
        self._origin = None
        self._positions = None
        if atoms is not None:
            self.set_atoms(atoms, origin=origin, extent=extent)

        if num_slices is None:
            self._slice_thickness = slice_thickness
        else:
            self._slice_thickness = self.thickness / num_slices

        self._periodic = periodic
        self._method = method
        self._atoms_per_loop = atoms_per_loop
        self._projector = PotentialProjector(num_samples=100)
        self._num_nodes = num_nodes

        if isinstance(parametrization, str):
            if parametrization == 'lobato':
                parametrization = LobatoPotential(tolerance=tolerance)

            elif parametrization == 'kirkland':
                parametrization = KirklandPotential(tolerance=tolerance)

            else:
                raise RuntimeError()

        self._parametrization = parametrization
        self._current_slice = 0

        if periodic is False:
            raise NotImplementedError()

        if method != 'splines':
            raise NotImplementedError()

    @property
    def atoms(self):
        return self._atoms

    @property
    def box(self):
        return np.concatenate((self._grid.extent, self.thickness[None]))

    @property
    def origin(self):
        return self._origin

    @property
    def thickness(self):
        return self.atoms.get_cell()[2, 2]

    @property
    def parametrization(self):
        return self._parametrization

    @property
    def num_slices(self):
        return np.ceil(self.thickness / self._slice_thickness).astype(np.int32)

    @property
    def slice_thickness(self):
        return self.thickness / self.num_slices

    @property
    def current_slice(self):
        return self._current_slice

    @current_slice.setter
    def current_slice(self, value):
        if value >= self.num_slices:
            raise RuntimeError('')

        self._current_slice = value

    @property
    def slice_entrance(self):
        return self.current_slice * self.slice_thickness

    @property
    def slice_exit(self):
        return (self.current_slice + 1) * self.slice_thickness

    def set_atoms(self, atoms, origin=None, extent=None):

        if atoms.cell[2, 2] == 0.:
            raise RuntimeError('atoms has no thickness')

        self._atoms = atoms
        self._species = list(np.unique(self.atoms.get_atomic_numbers()))
        self.new_view(origin=origin, extent=extent)

    def new_view(self, origin=None, extent=None):
        if origin is None:
            self._origin = np.zeros(2, dtype=np.float32)
        else:
            self._origin = np.array(origin, dtype=np.float32)

        if extent is None:
            self.extent = np.diag(self._atoms.get_cell())[:2]
        else:
            self.extent = np.array(extent, dtype=np.float32)

        self._positions = {}

    def get_positions(self, atomic_number):

        if atomic_number not in self._species:
            raise RuntimeError()

        if atomic_number not in self._positions.keys():
            self._update_view(atomic_number)

        return self._positions[atomic_number]

    def _update_view(self, atomic_number):

        def add_margin(positions, cell, extent, cutoff):

            def repeat_positions(positions, cell, n, axis):
                new_positions = positions.copy()
                for i in range(1, n):
                    new_positions[:, axis] += cell[axis]
                    positions = np.vstack((positions, new_positions))
                return positions

            for axis in [0, 1]:
                nrepeat = np.max((int((extent[axis] + 2 * cutoff) // cell[axis]), 1))
                positions = repeat_positions(positions, cell, nrepeat, axis)

                left_positions = positions[positions[:, axis] < (cutoff + extent[axis] - cell[axis] * nrepeat)]
                left_positions[:, axis] += cell[axis] * nrepeat

                right_positions = positions[(nrepeat * cell[axis] - positions[:, axis]) < cutoff]
                right_positions[:, axis] -= nrepeat * cell[axis]

                positions = np.concatenate((positions, left_positions, right_positions), axis=0)

            return positions

        positions = self.atoms.get_positions()[np.where(self.atoms.get_atomic_numbers() == atomic_number)[0]]
        positions[:, :2] = (positions[:, :2] - self.origin) % np.diag(self.atoms.get_cell())[:2]
        cutoff = self.parametrization.get_cutoff(atomic_number)

        positions = add_margin(positions, np.diag(self.atoms.get_cell())[:2], self.extent, cutoff)

        positions = positions[(positions[:, 0] < (self.extent[0] + cutoff)) &
                              (positions[:, 1] < (self.extent[1] + cutoff)), :]

        self._positions[atomic_number] = positions.astype(np.float32)

    def get_margin(self):
        cutoffs = [self.parametrization.get_cutoff(atomic_number) for atomic_number in self._species]
        return max(cutoffs)

    def get_positions_in_slice(self, atomic_number):
        positions = self.get_positions(atomic_number)
        cutoff = self.parametrization.get_cutoff(atomic_number)
        return positions[np.abs(self.slice_entrance + self.slice_thickness / 2 - positions[:, 2]) < (
                cutoff + self.slice_thickness / 2)]

    def slice_generator(self):
        for i in range(self.num_slices):
            self.current_slice = i
            yield self._calculate_tensor()

    def get_slice(self):
        return self.get_tensor()

    def get_tensor(self):
        return Tensor(self._calculate_tensor(), extent=self.extent, space='direct')

    def _calculate_tensor(self):
        self.check_is_defined()

        margin = (self.get_margin() / min(self.sampling)).astype(np.int32)
        padded_gpts = self.gpts + 2 * margin

        v = tf.Variable(tf.zeros(padded_gpts[0] * padded_gpts[1]))

        for atomic_number in self._species:
            positions = self.get_positions_in_slice(atomic_number)

            nodes = log_grid(min(self.sampling) / 2., self.parametrization.get_cutoff(atomic_number), self._num_nodes)

            if positions.shape[0] > 0:
                block_margin = tf.cast(self.parametrization.get_cutoff(atomic_number) / min(self.sampling),
                                       tf.int32)
                block_size = 2 * block_margin + 1

                x = tf.linspace(0., tf.cast(block_size, tf.float32) * self.sampling[0] - self.sampling[0],
                                block_size)[None, :]
                y = tf.linspace(0., tf.cast(block_size, tf.float32) * self.sampling[1] - self.sampling[1],
                                block_size)[None, :]

                block_indices = tf.range(0, block_size)[None, :] + \
                                tf.range(0, block_size)[:, None] * padded_gpts[1]

                a = self.slice_entrance - positions[:, 2]
                b = self.slice_exit - positions[:, 2]

                radials = self._projector.project(self.parametrization.get_soft_function(atomic_number), nodes, a, b)

                for start, size in batch_generator(positions.shape[0], self._atoms_per_loop):
                    batch_positions = tf.slice(positions, [start, 0], [size, -1])

                    corner_positions = tf.cast(tf.round(batch_positions[:, :2] / self.sampling),
                                               tf.int32) - block_margin + margin

                    block_positions = (batch_positions[:, :2] + self.sampling * margin
                                       - self.sampling * tf.cast(corner_positions, tf.float32))

                    batch_radials = tf.slice(radials, [start, 0], [size, -1])

                    r_interp = tf.reshape(tf.sqrt(((x - block_positions[:, 0][:, None]) ** 2)[:, :, None] +
                                                  ((y - block_positions[:, 1][:, None]) ** 2)[:, None, :]), (size, -1))

                    r_interp = tf.clip_by_value(r_interp, 0., tf.reduce_max(nodes))

                    v_interp = tf.Variable(tf.reshape(
                        interpolate_spline(
                            tf.tile(nodes[None, :], (size, 1))[:, :, None],
                            batch_radials[:, :, None],
                            r_interp[:, :, None], 1), (-1,)))

                    corner_indices = corner_positions[:, 0] * padded_gpts[1] + corner_positions[:, 1]
                    indices = tf.reshape(corner_indices[:, None, None] + block_indices[None, :, :], (-1, 1))
                    indices = tf.clip_by_value(indices, 0, v.shape[0] - 1)

                    tf.compat.v1.scatter_nd_add(v, indices, v_interp)

        v = tf.reshape(v, padded_gpts)
        v = tf.slice(v, (margin, margin), self.gpts)

        return v[None, :, :] / kappa

    def show(self, **kwargs):
        self.get_tensor().show(**kwargs)

    def show_atoms(self, plane='xy', scale=100, margin=True, fig_scale=1):

        from tensorwaves.plotutils import display_atoms

        box = self.box
        origin = np.array([0., 0., 0.])

        positions = []
        atomic_numbers = []

        for atomic_number in self._species:
            positions.append(self.get_positions(atomic_number))
            atomic_numbers.append([atomic_number] * len(self.get_positions(atomic_number)))

        display_atoms(np.vstack(positions), np.hstack(atomic_numbers).astype(np.int), plane=plane, origin=origin,
                      box=box, scale=scale, fig_scale=fig_scale)

    def to_array(self):

        new_array = np.zeros(tuple(np.append(self.gpts, self.num_slices)), dtype=np.float32)

        for i, tensor in enumerate(self.slice_generator()):
            new_array[..., i] = tensor

        box = np.append(self.extent, self.thickness)

        return ArrayPotential(array=new_array, box=box, num_slices=self.num_slices, projected=True)


def next_divisor(n, m):
    if n % m:
        if m > n // 2:
            raise RuntimeError()
        m += 1
        return next_divisor(n, m)
    else:
        return m


class ArrayPotential(HasGrid):

    def __init__(self, array, box, num_slices=None, projected=False):

        self._array = np.float32(array - array.min())

        if num_slices is None:
            num_slices = np.ceil(box[2] / .5).astype(np.int32)

        if array.shape[2] % num_slices:
            num_slices = next_divisor(array.shape[2], num_slices)

        extent = box[:2]
        gpts = GridProperty(lambda: np.array([dim for dim in self._array.shape[:2]]), dtype=np.int32, locked=True)

        HasGrid.__init__(self, extent=extent, gpts=gpts)

        self._thickness = box[2]
        self._num_slices = num_slices
        self._projected = projected

    def numpy(self):
        return self._array

    def repeat(self, multiples):
        self._array = tf.tile(self._array, multiples)
        self.extent = multiples[:2] * self.extent
        self._thickness = multiples[2] * self._thickness

    def set_view(self, origin, extent):

        start = np.round(origin / self.extent * self.gpts).astype(np.int32)
        origin = start * self.sampling

        end = np.round((origin + extent) / self.extent * self.gpts).astype(np.int32)
        repeat = np.ceil(end / self.gpts.astype(np.float)).astype(int)

        new_array = np.tile(self._array, np.append(repeat, 1))

        self._array = new_array[start[0]:end[0], start[1]:end[1], :]

        self.sampling = self.sampling

    @property
    def box(self):
        return np.concatenate((self.extent, [self.thickness]))

    @property
    def voxel_height(self):
        return self.thickness / self._array.shape[2]

    @property
    def slice_thickness_voxels(self):
        return self._array.shape[2] // self.num_slices

    @property
    def thickness(self):
        return self._thickness

    @property
    def num_slices(self):
        return self._num_slices

    @property
    def slice_thickness(self):
        return self.thickness / self.num_slices

    def downsample(self):
        N, M = self.gpts

        X = np.fft.fft(self._array, axis=0)
        self._array = np.fft.ifft((X[:(N // 2), :] + X[-(N // 2):, :]) / 2., axis=0).real

        X = np.fft.fft(self._array, axis=1)
        self._array = np.fft.ifft((X[:, :(M // 2)] + X[:, -(M // 2):]) / 2., axis=1).real

        self.extent = self.extent

    def project(self):
        if self._projected:
            raise RuntimeError()

        new_array = np.zeros(tuple(np.append(self.gpts, self.num_slices)), dtype=np.float32)

        for i, tensor in enumerate(self.slice_generator()):
            new_array[..., i] = tensor

        self._array = new_array

        self._projected = True

    def slice_generator(self):
        for i in range(self.num_slices):
            yield self._create_tensor(i)

    def get_tensor(self, i=0):
        return Tensor(self._create_tensor(i), extent=self.extent)

    def _create_tensor(self, i=None):

        if self._projected:
            return tf.convert_to_tensor(self._array[..., i][None, :, :], dtype=tf.float32)

        return (tf.reduce_sum(self._array[:, :, i * self.slice_thickness_voxels:
                                                i * self.slice_thickness_voxels + self.slice_thickness_voxels],
                              axis=2) * self.voxel_height)[None, :, :]

    def copy(self):
        box = np.append(self.extent, self.thickness)
        return self.__class__(array=self._array.copy(), box=box, num_slices=self._num_slices, projected=self._projected)


def gaussian(rgd, alpha):
    r_g = rgd.r_g
    g_g = 4 / np.sqrt(np.pi) * alpha ** 1.5 * np.exp(-alpha * r_g ** 2)
    return g_g


class Interpolator:
    def __init__(self, gd1, gd2, dtype=float):
        from gpaw.wavefunctions.pw import PWDescriptor
        self.pd1 = PWDescriptor(0.0, gd1, dtype)
        self.pd2 = PWDescriptor(0.0, gd2, dtype)

    def interpolate(self, a_r):
        return self.pd1.interpolate(a_r, self.pd2)[0]


def potential_from_GPAW(calc, h=.05, rcgauss=0.02, spline_pts=200, n=2):
    from gpaw.lfc import LFC
    from gpaw.utilities import h2gpts
    from gpaw.fftw import get_efficient_fft_size
    from gpaw.grid_descriptor import GridDescriptor
    from gpaw.mpi import serial_comm

    v_r = calc.get_electrostatic_potential() / units.Ha

    old_gd = GridDescriptor(calc.hamiltonian.finegd.N_c, calc.hamiltonian.finegd.cell_cv, comm=serial_comm)

    N_c = h2gpts(h / units.Bohr, calc.wfs.gd.cell_cv)
    N_c = np.array([get_efficient_fft_size(N, n) for N in N_c])
    new_gd = GridDescriptor(N_c, calc.wfs.gd.cell_cv, comm=serial_comm)

    interpolator = Interpolator(old_gd, new_gd)

    v_r = interpolator.interpolate(v_r)

    dens = calc.density
    dens.D_asp.redistribute(dens.atom_partition.as_serial())
    dens.Q_aL.redistribute(dens.atom_partition.as_serial())

    alpha = 1 / (rcgauss / units.Bohr) ** 2
    dv_a1 = []
    for a, D_sp in dens.D_asp.items():
        setup = dens.setups[a]
        c = setup.xc_correction
        rgd = c.rgd
        ghat_g = gaussian(rgd, 1 / setup.rcgauss ** 2)
        Z_g = gaussian(rgd, alpha) * setup.Z
        D_q = np.dot(D_sp.sum(0), c.B_pqL[:, :, 0])
        dn_g = np.dot(D_q, (c.n_qg - c.nt_qg)) * np.sqrt(4 * np.pi)
        dn_g += 4 * np.pi * (c.nc_g - c.nct_g)
        dn_g -= Z_g
        dn_g -= dens.Q_aL[a][0] * ghat_g * np.sqrt(4 * np.pi)
        dv_g = rgd.poisson(dn_g) / np.sqrt(4 * np.pi)
        dv_g[1:] /= rgd.r_g[1:]
        dv_g[0] = dv_g[1]
        dv_g[-1] = 0.0
        dv_a1.append([rgd.spline(dv_g, points=spline_pts)])

    dens.D_asp.redistribute(dens.atom_partition)
    dens.Q_aL.redistribute(dens.atom_partition)

    if dv_a1:
        dv = LFC(new_gd, dv_a1)
        dv.set_positions(calc.spos_ac)
        dv.add(v_r)
    dens.gd.comm.broadcast(v_r, 0)

    return ArrayPotential(array=-v_r * units.Ha, box=np.diag(calc.atoms.cell))
