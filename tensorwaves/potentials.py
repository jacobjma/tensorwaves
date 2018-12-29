import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from scipy.optimize import brentq

from tensorwaves.bases import Tensor, HasData, HasGrid
from tensorwaves.utils import kappa, batch_generator


class ParameterizedPotential(object):

    def __init__(self, filename=None, parameters=None, tolerance=1e-2):

        super().__init__()

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
        super().__init__(filename='data/lobato.txt', tolerance=tolerance)

    def _create_function(self, atomic_number):
        a = [np.float32(np.pi ** 2 * self.parameters[atomic_number][key_a] /
                        self.parameters[atomic_number][key_b] ** (3 / 2.))
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]

        b = [np.float32(2 * np.pi / tf.sqrt(self.parameters[atomic_number][key])) for key in
             ('b1', 'b2', 'b3', 'b4', 'b5')]

        def func(r):
            return (a[0] * (2. / (b[0] * r) + 1) * tf.exp(-b[0] * r) +
                    a[1] * (2. / (b[1] * r) + 1) * tf.exp(-b[1] * r) +
                    a[2] * (2. / (b[2] * r) + 1) * tf.exp(-b[2] * r) +
                    a[3] * (2. / (b[3] * r) + 1) * tf.exp(-b[3] * r) +
                    a[4] * (2. / (b[4] * r) + 1) * tf.exp(-b[4] * r))

        return func

    def _create_soft_function(self, atomic_number, r_cut):
        a = [np.pi ** 2 * self.parameters[atomic_number][key_a] / self.parameters[atomic_number][key_b] ** (3 / 2.)
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
        b = [2 * np.pi / tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

        dvdr_cut = - (a[0] * (2 / (b[0] * r_cut ** 2) + 2 / r_cut + b[0]) * tf.exp(-b[0] * r_cut) +
                      a[1] * (2 / (b[1] * r_cut ** 2) + 2 / r_cut + b[1]) * tf.exp(-b[1] * r_cut) +
                      a[2] * (2 / (b[2] * r_cut ** 2) + 2 / r_cut + b[2]) * tf.exp(-b[2] * r_cut) +
                      a[3] * (2 / (b[3] * r_cut ** 2) + 2 / r_cut + b[3]) * tf.exp(-b[3] * r_cut) +
                      a[4] * (2 / (b[4] * r_cut ** 2) + 2 / r_cut + b[4]) * tf.exp(-b[4] * r_cut))

        v_cut = (a[0] * (2. / (b[0] * r_cut) + 1) * tf.exp(-b[0] * r_cut) +
                 a[1] * (2. / (b[1] * r_cut) + 1) * tf.exp(-b[1] * r_cut) +
                 a[2] * (2. / (b[2] * r_cut) + 1) * tf.exp(-b[2] * r_cut) +
                 a[3] * (2. / (b[3] * r_cut) + 1) * tf.exp(-b[3] * r_cut) +
                 a[4] * (2. / (b[4] * r_cut) + 1) * tf.exp(-b[4] * r_cut))

        def func(r):
            v = (a[0] * (2. / (b[0] * r) + 1) * tf.exp(-b[0] * r) +
                 a[1] * (2. / (b[1] * r) + 1) * tf.exp(-b[1] * r) +
                 a[2] * (2. / (b[2] * r) + 1) * tf.exp(-b[2] * r) +
                 a[3] * (2. / (b[3] * r) + 1) * tf.exp(-b[3] * r) +
                 a[4] * (2. / (b[4] * r) + 1) * tf.exp(-b[4] * r))

            return v - v_cut - (r - r_cut) * dvdr_cut

        return func

    def _create_projected_function(self, atomic_number):
        


class KirklandPotential(ParameterizedPotential):

    def __init__(self, tolerance=1e-2):
        super().__init__(filename='data/kirland.txt', tolerance=tolerance)

    def _create_function(self, atomic_number):
        a = [np.pi * self.parameters[atomic_number][key] for key in ('a1', 'a2', 'a3')]
        b = [2 * np.pi * tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3')]
        c = [np.pi ** (3 / 2.) * self.parameters[atomic_number][key_c] / self.parameters[atomic_number][key_d] ** (
                3 / 2.) for key_c, key_d in
             zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))]
        d = [np.pi ** 2 / self.parameters[atomic_number][key] for key in ('d1', 'd2', 'd3')]

        def func(r):
            return (a[0] * tf.exp(-b[0] * r) / r + c[0] * tf.exp(-d[0] * r ** 2) +
                    a[1] * tf.exp(-b[1] * r) / r + c[1] * tf.exp(-d[1] * r ** 2) +
                    a[2] * tf.exp(-b[2] * r) / r + c[2] * tf.exp(-d[2] * r ** 2))

        # dvdr = lambda r: (- a[0] * (1 / r + b[0]) * tf.exp(-b[0] * r) / r - 2 * c[0] * d[0] * r * tf.exp(-d[0] * r ** 2)
        #                   - a[1] * (1 / r + b[1]) * tf.exp(-b[1] * r) / r - 2 * c[1] * d[1] * r * tf.exp(-d[1] * r ** 2)
        #                   - a[2] * (1 / r + b[2]) * tf.exp(-b[2] * r) / r - 2 * c[2] * d[2] * r * tf.exp(
        #             -d[2] * r ** 2))
        return func

    def _create_soft_function(self, atomic_number, r_cut):
        pass


class Quadrature(object):

    def __init__(self, num_samples=25, num_nodes=100):
        self._num_nodes = num_nodes

        self._quadrature = {}
        self._quadrature['xk'] = np.linspace(-1., 1., num_samples + 1).astype(np.float32)
        self._quadrature['wk'] = self._quadrature['xk'][1:] - self._quadrature['xk'][:-1]
        self._quadrature['xk'] = self._quadrature['xk'][:-1]

    @property
    def num_nodes(self):
        return self._num_nodes

    def get_integrals(self, function, a, b, nodes):

        xkab = self._quadrature['xk'][None, :] * ((b - a) / 2)[:, None] + ((a + b) / 2)[:, None]
        wkab = self._quadrature['wk'][None, :] * ((b - a) / 2)[:, None]
        #print(a, b, xkab, wkab)
        r = tf.sqrt(nodes[None, None, :] ** 2 + (xkab ** 2)[:, :, None])
        r = tf.clip_by_value(r, 0, nodes[-1])
        return tf.reduce_sum(function(r) * wkab[:, :, None], axis=1)


class Potential(HasData, HasGrid):

    def __init__(self, atoms=None, origin=None, extent=None, gpts=None, sampling=None, slice_thickness=.5,
                 parametrization='lobato', periodic=True, method='splines', atoms_per_loop=10, save_data=True,
                 grid=None):
        HasData.__init__(self, save_data=save_data)
        HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, space='direct', grid=grid)

        if isinstance(parametrization, str):
            if parametrization.lower() == 'lobato':
                self._parametrization = LobatoPotential()
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
        return Tensor(self._create_tensor(i), extent=self.grid.extent, space=self.grid.space)

    def _create_tensor(self, i=None):
        margin = (self.get_margin() / min(self.grid.sampling)).astype(np.int32)
        padded_gpts = self.grid.gpts + 2 * margin

        v = tf.Variable(tf.zeros(np.prod(padded_gpts)))

        for atomic_number in self._unique_atomic_numbers:
            positions = self.get_positions_in_slice(atomic_number, i)

            def create_nodes(r_min, r_cut):
                dt = np.log(r_cut / r_min) / (self._quadrature.num_nodes - 1)
                return (r_min * np.exp(dt * np.linspace(0., self._quadrature.num_nodes - 1,
                                                        self._quadrature.num_nodes))).astype(np.float32)

            nodes = create_nodes(min(self.grid.sampling), self.parametrization.get_cutoff(atomic_number))

            if positions.shape[0] > 0:
                block_margin = tf.cast(self.parametrization.get_cutoff(atomic_number) / min(self.grid.sampling),
                                       tf.int32)
                block_size = 2 * block_margin + 1

                x = tf.linspace(0., tf.cast(block_size, tf.float32) * self.grid.sampling[0], block_size)[None, :]
                y = tf.linspace(0., tf.cast(block_size, tf.float32) * self.grid.sampling[1], block_size)[None, :]

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
                    # print(r_interp)
                    # import matplotlib.pyplot as plt
                    # plt.plot(r_interp[0,:].numpy())
                    # plt.plot(batch_radials[0].numpy())
                    # plt.show()

                    # sss

                    v_interp = tf.Variable(tf.reshape(
                        tf.contrib.image.interpolate_spline(
                            tf.tile(nodes[None, :], (size, 1))[:, :, None],
                            batch_radials[:, :, None],
                            r_interp[:, :, None], 1), (-1,)))

                    corner_indices = corner_positions[:, 0] * padded_gpts[1] + corner_positions[:, 1]
                    indices = tf.reshape(corner_indices[:, None, None] + block_indices[None, :, :], (-1, 1))

                    tf.scatter_nd_add(v, indices, v_interp)

        v = tf.reshape(v, padded_gpts)
        v = tf.slice(v, (margin, margin), self.grid.gpts)

        return v[None, :, :] / kappa

    def show(self, i=0, mode='magnitude', space='direct', color_scale='linear', **kwargs):
        self.get_tensor(i).show(mode=mode, space=space, **kwargs)

    def show_atoms(self, plane='xy', i=None):

        box = self.box
        origin = np.array([0., 0., 0.])

        positions = []
        atomic_numbers = []
        if i is None:
            for atomic_number in np.unique(self.atoms.get_atomic_numbers()):
                positions.append(self.get_positions(atomic_number))
                atomic_numbers.append([atomic_number] * len(self.get_positions(atomic_number)))

        else:
            box[2] = self.slice_thickness
            origin[2] = self.slice_entrance(i)

            for atomic_number in np.unique(self.atoms.get_atomic_numbers()):
                positions.append(self.get_positions_in_slice(atomic_number, i))
                atomic_numbers.append([atomic_number] * len(self.get_positions_in_slice(atomic_number, i)))

        display_atoms(np.vstack(positions), np.hstack(atomic_numbers).astype(np.int), plane=plane, origin=origin,
                      box=box)


def plane2axes(plane):
    axes = ()
    for axis in list(plane):
        if axis == 'x': axes += (0,)
        if axis == 'y': axes += (1,)
        if axis == 'z': axes += (2,)
    return axes


def display_atoms(positions, numbers, plane, origin, box, scale=100, ax=None, colors=None):
    if ax is None:
        fig, ax = plt.subplots()

    axes = plane2axes(plane)
    edges = np.zeros((2, 5))
    edges[0, :] += origin[axes[0]]
    edges[1, :] += origin[axes[1]]
    edges[0, 2:4] += np.array([box[0], box[1], box[2]])[axes[0]]
    edges[1, 1:3] += np.array([box[0], box[1], box[2]])[axes[1]]

    ax.plot(*edges, 'k-')

    if len(positions) > 0:
        positions = positions[:, axes]
        if colors is None:
            colors = cpk_colors[numbers]
        sizes = covalent_radii[numbers]

        ax.scatter(*positions.T, c=colors, s=scale * sizes)
        ax.axis('equal')

    plt.show()
