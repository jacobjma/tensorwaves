import csv
import os

import numpy as np
import tensorflow as tf

from scipy.optimize import brentq

from tensorwaves.bases import Tensor, HasData, Showable, GridProperty
from tensorwaves.utils import kappa_ASE, batch_generator


class ParameterizedPotential(object):

    def __init__(self, filename=None, parameters=None, tolerance=1e-2):

        self._tolerance = tolerance
        self._cutoffs = {}
        self._functions = {}
        self._soft_functions = {}

        self.filename = filename
        if parameters is None:
            self.parameters = self.load_parameters(self.filename)

    @property
    def tolerance(self):
        return self._tolerance

    def load_parameters(self, filename):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        parameters = {}
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            keys = next(reader)
            for _, row in enumerate(reader):
                values = list(map(float, row))
                parameters[int(row[0])] = dict(zip(keys, values))
        return parameters

    def _create_projected_function(self, atomic_number):
        raise RuntimeError()

    def _create_function(self, atomic_number):
        raise RuntimeError()

    def _create_soft_function(self, atomic_number, r_cut):
        raise RuntimeError()

    def _find_cutoff(self, atomic_number):
        return np.float32(brentq(lambda x: self.get_function(atomic_number)(x) - self.tolerance, 1e-16, 100))

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


class LobatoPotential(ParameterizedPotential):

    def __init__(self, tolerance=1e-2):
        ParameterizedPotential.__init__(self, filename='data/lobato.txt', tolerance=tolerance)

    def _create_function(self, atomic_number):
        a = [np.float32(np.pi ** 2 * self.parameters[atomic_number][key_a] /
                        self.parameters[atomic_number][key_b] ** (3 / 2.))
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]

        b = [2 * np.pi / tf.sqrt(self.parameters[atomic_number][key]) for key in
             ('b1', 'b2', 'b3', 'b4', 'b5')]

        def func(r):
            return (a[0] * (2. / (b[0] * r) + 1.) * tf.exp(-b[0] * r) +
                    a[1] * (2. / (b[1] * r) + 1.) * tf.exp(-b[1] * r) +
                    a[2] * (2. / (b[2] * r) + 1.) * tf.exp(-b[2] * r) +
                    a[3] * (2. / (b[3] * r) + 1.) * tf.exp(-b[3] * r) +
                    a[4] * (2. / (b[4] * r) + 1.) * tf.exp(-b[4] * r))

        return func

    def _create_soft_function(self, atomic_number, r_cut):
        a = [np.pi ** 2 * self.parameters[atomic_number][key_a] / self.parameters[atomic_number][key_b] ** (3 / 2.)
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
        b = [2 * np.pi / tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

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
            v = (a[0] * (2. / (b[0] * r) + 1.) * tf.exp(-b[0] * r) +
                 a[1] * (2. / (b[1] * r) + 1.) * tf.exp(-b[1] * r) +
                 a[2] * (2. / (b[2] * r) + 1.) * tf.exp(-b[2] * r) +
                 a[3] * (2. / (b[3] * r) + 1.) * tf.exp(-b[3] * r) +
                 a[4] * (2. / (b[4] * r) + 1.) * tf.exp(-b[4] * r))

            return v - v_cut - (r - r_cut) * dvdr_cut

        return func

    def _create_projected_function(self, atomic_number):
        pass


class KirklandPotential(ParameterizedPotential):

    def __init__(self, tolerance=1e-3):
        ParameterizedPotential.__init__(self, filename='data/kirland.txt', tolerance=tolerance)

    def _create_function(self, atomic_number):
        a = [np.pi * self.parameters[atomic_number][key] for key in ('a1', 'a2', 'a3')]
        b = [2. * np.pi * tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3')]
        c = [np.pi ** (3. / 2.) * self.parameters[atomic_number][key_c] / self.parameters[atomic_number][key_d] ** (
                3. / 2.) for key_c, key_d in
             zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))]
        d = [np.pi ** 2 / self.parameters[atomic_number][key] for key in ('d1', 'd2', 'd3')]

        def func(r):
            return (a[0] * tf.exp(-b[0] * r) / r + c[0] * tf.exp(-d[0] * r ** 2.) +
                    a[1] * tf.exp(-b[1] * r) / r + c[1] * tf.exp(-d[1] * r ** 2.) +
                    a[2] * tf.exp(-b[2] * r) / r + c[2] * tf.exp(-d[2] * r ** 2.))

        # dvdr = lambda r: (- a[0] * (1 / r + b[0]) * tf.exp(-b[0] * r) / r - 2 * c[0] * d[0] * r * tf.exp(-d[0] * r ** 2)
        #                   - a[1] * (1 / r + b[1]) * tf.exp(-b[1] * r) / r - 2 * c[1] * d[1] * r * tf.exp(-d[1] * r ** 2)
        #                   - a[2] * (1 / r + b[2]) * tf.exp(-b[2] * r) / r - 2 * c[2] * d[2] * r * tf.exp(
        #             -d[2] * r ** 2))
        return func

    def _create_soft_function(self, atomic_number, r_cut):
        pass


class Quadrature(object):

    def __init__(self, num_samples=200, num_nodes=100):
        self._num_nodes = num_nodes

        self._quadrature = {}
        self._quadrature['xk'] = np.linspace(-1., 1., num_samples + 1).astype(np.float32)
        self._quadrature['wk'] = self._quadrature['xk'][1:] - self._quadrature['xk'][:-1]
        self._quadrature['xk'] = self._quadrature['xk'][:-1]

    @property
    def num_nodes(self):
        return self._num_nodes

    def get_integrals(self, function, a, b, nodes):
        xkab = self._quadrature['xk'][None, :] * ((b - a) / 2.)[:, None] + ((a + b) / 2.)[:, None]
        wkab = self._quadrature['wk'][None, :] * ((b - a) / 2.)[:, None]

        r = tf.sqrt(nodes[None, None, :] ** 2. + (xkab ** 2.)[:, :, None])
        r = tf.clip_by_value(r, 0, nodes[-1])
        return tf.reduce_sum(function(r) * wkab[:, :, None], axis=1)


class Potential(HasData, Showable):

    def __init__(self, atoms=None, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5,
                 parametrization='lobato', periodic=True, method='splines', atoms_per_loop=10, tolerance=1e-2,
                 save_data=True, grid=None):

        HasData.__init__(self, save_data=save_data)
        Showable.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid, space='direct')

        if isinstance(parametrization, str):
            if parametrization.lower() == 'lobato':
                self._parametrization = LobatoPotential(tolerance=tolerance)
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()

        self.set_atoms(atoms, origin=origin, extent=extent)

        self._slice_thickness = slice_thickness

        self._quadrature = Quadrature()
        self._atoms_per_loop = atoms_per_loop

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
        return np.ceil(self.thickness / self.slice_thickness).astype(np.int32)

    @property
    def slice_thickness(self):
        return self._slice_thickness

    def slice_entrance(self, i):
        return i * self.slice_thickness

    def slice_exit(self, i):
        return (i + 1) * self.slice_thickness

    def set_atoms(self, atoms, origin=None, extent=None):
        self._atoms = atoms

        self._unique_atomic_numbers = np.unique(self.atoms.get_atomic_numbers())

        self.set_view(origin=origin, extent=extent)

    def set_view(self, origin=None, extent=None):
        if origin is None:
            self._origin = np.zeros(2, dtype=np.float32)
        else:
            self._origin = np.array(origin, dtype=np.float32)

        if extent is None:
            self._grid.extent = np.diag(self.atoms.get_cell())[:2]
        else:
            self._origin = np.array(extent, dtype=np.float32)

        self._positions = {}

    def get_margin(self):
        cutoffs = [self.parametrization.get_cutoff(atomic_number) for atomic_number in self._unique_atomic_numbers]
        return max(cutoffs)

    def get_positions(self, atomic_number):
        try:
            return self._positions[atomic_number]
        except KeyError:
            self._positions[atomic_number] = (self._update_view(atomic_number)).astype(np.float32)
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

        new_positions = np.zeros((0, 3))
        positions = self.atoms.get_positions()[np.where(self.atoms.get_atomic_numbers() == atomic_number)[0]]
        positions[:, :2] = (positions[:, :2] - self.origin) % np.diag(self.atoms.get_cell())[:2]
        cutoff = self.parametrization.get_cutoff(atomic_number)

        positions = add_margin(positions, np.diag(self.atoms.get_cell())[:2],
                               self.grid.extent, cutoff)

        positions = positions[(positions[:, 0] < (self.grid.extent[0] + cutoff)) &
                              (positions[:, 1] < (self.grid.extent[1] + cutoff)), :]

        return np.concatenate((new_positions, positions))

    def get_positions_in_slice(self, atomic_number, i):
        positions = self.get_positions(atomic_number)
        cutoff = self.parametrization.get_cutoff(atomic_number)
        return positions[np.abs(self.slice_entrance(i) + self.slice_thickness / 2 - positions[:, 2]) < (
                cutoff + self.slice_thickness / 2)]

    def slice_generator(self):
        for i in range(self.num_slices):
            yield self._create_tensor(i)

    def get_tensor(self, i=0):
        return Tensor(self._create_tensor(i), extent=self.grid.extent, space=self.space)

    def _create_tensor(self, i=None):
        margin = (self.get_margin() / min(self.grid.sampling)).astype(np.int32)
        padded_gpts = self.grid.gpts + 2 * margin

        v = tf.contrib.eager.Variable(tf.zeros(np.prod(padded_gpts)))

        for atomic_number in self._unique_atomic_numbers:
            positions = self.get_positions_in_slice(atomic_number, i)

            def create_nodes(r_min, r_cut):
                dt = np.log(r_cut / r_min) / (self._quadrature.num_nodes - 1.)
                return (r_min * np.exp(dt * np.linspace(0., self._quadrature.num_nodes - 1.,
                                                        self._quadrature.num_nodes))).astype(np.float32)

            nodes = create_nodes(min(self.grid.sampling) / 2., self.parametrization.get_cutoff(atomic_number))

            if positions.shape[0] > 0:
                block_margin = tf.cast(self.parametrization.get_cutoff(atomic_number) / min(self.grid.sampling),
                                       tf.int32)
                block_size = 2 * block_margin + 1

                x = tf.linspace(0., tf.cast(block_size, tf.float32) * self.grid.sampling[0] - self.grid.sampling[0],
                                block_size)[None, :]
                y = tf.linspace(0., tf.cast(block_size, tf.float32) * self.grid.sampling[1] - self.grid.sampling[1],
                                block_size)[None, :]

                block_indices = tf.range(0, block_size)[None, :] + \
                                tf.range(0, block_size)[:, None] * padded_gpts[1]

                a = self.slice_entrance(i) - positions[:, 2]
                b = self.slice_exit(i) - positions[:, 2]

                radials = self._quadrature.get_integrals(self.parametrization.get_soft_function(atomic_number), a, b,
                                                         nodes)

                for start, size in batch_generator(positions.shape[0], self._atoms_per_loop):
                    batch_positions = tf.slice(positions, [start, 0], [size, -1])

                    corner_positions = tf.cast(tf.round(batch_positions[:, :2] / self.grid.sampling),
                                               tf.int32) - block_margin + margin

                    block_positions = (batch_positions[:, :2] + self.grid.sampling * margin
                                       - self.grid.sampling * tf.cast(corner_positions, tf.float32))

                    batch_radials = tf.slice(radials, [start, 0], [size, -1])

                    r_interp = tf.reshape(tf.sqrt(((x - block_positions[:, 0][:, None]) ** 2)[:, :, None] +
                                                  ((y - block_positions[:, 1][:, None]) ** 2)[:, None, :]), (size, -1))

                    r_interp = tf.clip_by_value(r_interp, 0., tf.reduce_max(nodes))

                    v_interp = tf.contrib.eager.Variable(tf.reshape(
                        tf.contrib.image.interpolate_spline(
                            tf.tile(nodes[None, :], (size, 1))[:, :, None],
                            batch_radials[:, :, None],
                            r_interp[:, :, None], 1), (-1,)))

                    corner_indices = corner_positions[:, 0] * padded_gpts[1] + corner_positions[:, 1]
                    indices = tf.reshape(corner_indices[:, None, None] + block_indices[None, :, :], (-1, 1))
                    indices = tf.clip_by_value(indices, 0, v.shape[0] - 1)

                    tf.scatter_nd_add(v, indices, v_interp)

        v = tf.reshape(v, padded_gpts)
        v = tf.slice(v, (margin, margin), self.grid.gpts)

        return v[None, :, :] / kappa_ASE

    def get_showable_tensor(self, i=0):
        return self.get_tensor(i=i)

    # def show(self, i=0, mode='magnitude', space='direct', color_scale='linear', **kwargs):

    # self.get_tensor(i).show(mode=mode, space=space, **kwargs)

    def show_atoms(self, plane='xy', scale=100, i=None, margin=True, fig_scale=1):

        from tensorwaves.plotutils import display_atoms

        box = self.box
        origin = np.array([0., 0., 0.])

        if margin:
            atoms = self
        else:
            atoms = self.atoms

        positions = []
        atomic_numbers = []
        if i is None:
            for atomic_number in np.unique(self.atoms.get_atomic_numbers()):
                positions.append(atoms.get_positions(atomic_number))
                atomic_numbers.append([atomic_number] * len(atoms.get_positions(atomic_number)))

        else:
            box[2] = self.slice_thickness
            origin[2] = self.slice_entrance(i)

            for atomic_number in np.unique(self.atoms.get_atomic_numbers()):
                positions.append(self.get_positions_in_slice(atomic_number, i))
                atomic_numbers.append([atomic_number] * len(self.get_positions_in_slice(atomic_number, i)))

        display_atoms(np.vstack(positions), np.hstack(atomic_numbers).astype(np.int), plane=plane, origin=origin,
                      box=box, scale=scale, fig_scale=fig_scale)

    def to_array(self):

        new_array = np.zeros(tuple(np.append(self.grid.gpts, self.num_slices)), dtype=np.float32)

        for i, tensor in enumerate(self.slice_generator()):
            new_array[..., i] = tensor

        box = np.append(self.grid.extent, self.thickness)

        return ArrayPotential(array=new_array, box=box, num_slices=self.num_slices, projected=True)


class ArrayPotential(Showable):

    def __init__(self, array, box, num_slices=None, projected=False):

        self._array = np.float32(array - array.min())

        if num_slices is None:
            num_slices = np.ceil(box[2] / .5).astype(np.int32)

        if array.shape[2] % num_slices:
            raise RuntimeError()

        extent = box[:2]
        gpts = GridProperty(lambda: np.array([dim for dim in self._array.shape[:2]]), dtype=np.int32, locked=True)

        Showable.__init__(self, extent=extent, gpts=gpts, space='direct')

        self._thickness = box[2]
        self._num_slices = num_slices
        self._projected = projected

    def repeat(self, multiples):
        self._array = tf.tile(self._array, multiples)
        self.grid.extent = multiples[:2] * self.grid.extent
        self._thickness = multiples[2] * self._thickness

    def set_view(self, origin, extent):

        start = np.round(origin / self.grid.extent * self.grid.gpts).astype(np.int32)
        origin = start * self.grid.sampling

        end = np.round((origin + extent) / self.grid.extent * self.grid.gpts).astype(np.int32)
        repeat = np.ceil(end / self.grid.gpts.astype(np.float)).astype(int)

        new_array = np.tile(self._array, np.append(repeat, 1))

        self._array = new_array[start[0]:end[0], start[1]:end[1], :]

        self.grid.sampling = self.grid.sampling

    @property
    def box(self):
        return np.concatenate((self.grid.extent, [self.thickness]))

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
        N, M = self.grid.gpts

        X = np.fft.fft(self._array, axis=0)
        self._array = np.fft.ifft((X[:(N // 2), :] + X[-(N // 2):, :]) / 2., axis=0).real

        X = np.fft.fft(self._array, axis=1)
        self._array = np.fft.ifft((X[:, :(M // 2)] + X[:, -(M // 2):]) / 2., axis=1).real

        self.grid.extent = self.grid.extent

    def project(self):
        if self._projected:
            raise RuntimeError()

        new_array = np.zeros(tuple(np.append(self.grid.gpts, self.num_slices)), dtype=np.float32)

        for i, tensor in enumerate(self.slice_generator()):
            new_array[..., i] = tensor

        self._array = new_array

        self._projected = True

    def slice_generator(self):
        for i in range(self.num_slices):
            yield self._create_tensor(i)

    def get_tensor(self, i=0):
        return Tensor(self._create_tensor(i), extent=self.grid.extent, space=self.space)

    def _create_tensor(self, i=None):

        if self._projected:
            return tf.convert_to_tensor(self._array[..., i][None, :, :], dtype=tf.float32)

        return (tf.reduce_sum(self._array[:, :, i * self.slice_thickness_voxels:
                                                i * self.slice_thickness_voxels + self.slice_thickness_voxels],
                              axis=2) * self.voxel_height)[None, :, :]

    def copy(self):
        box = np.append(self.grid.extent, self.thickness)
        return self.__class__(array=self._array.copy(), box=box, num_slices=self._num_slices, projected=self._projected)


def potential_from_GPAW(calc):
    from gpaw.utilities.ps2ae import PS2AE
    potential_array = -PS2AE(calc).get_electrostatic_potential(ae=True)
    return ArrayPotential(array=potential_array, box=np.diag(calc.atoms.cell))
