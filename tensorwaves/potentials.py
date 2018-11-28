import csv
import math
import os
from collections import OrderedDict

import ipywidgets
import numpy as np
import tensorflow as tf
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from scipy.optimize import brentq

from tensorwaves.bases import Grid
import matplotlib.pyplot as plt


# from tensorwaves.display import Display


class PotentialType(object):

    def __init__(self, filename='data/lobato.txt', parameters=None):
        self.filename = filename
        if parameters is None:
            self.parameters = self.load_parameters(self.filename)

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

    def get_projected_potential_function(self, atomic_number):
        return None

    def get_potential_function(self, atomic_number):
        return None

    def get_soft_potential_function(self, atomic_number, r_cut):
        return None


class LobatoPotential(PotentialType):

    def __init__(self):
        PotentialType.__init__(self)

    def get_potential_function(self, atomic_number):
        a = [np.pi ** 2 * self.parameters[atomic_number][key_a] / self.parameters[atomic_number][key_b] ** (3 / 2.)
             for key_a, key_b in zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
        b = [2 * np.pi / tf.sqrt(self.parameters[atomic_number][key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

        def func(r):
            return (a[0] * (2. / (b[0] * r) + 1) * tf.exp(-b[0] * r) +
                    a[1] * (2. / (b[1] * r) + 1) * tf.exp(-b[1] * r) +
                    a[2] * (2. / (b[2] * r) + 1) * tf.exp(-b[2] * r) +
                    a[3] * (2. / (b[3] * r) + 1) * tf.exp(-b[3] * r) +
                    a[4] * (2. / (b[4] * r) + 1) * tf.exp(-b[4] * r))

        return func

    def get_soft_potential_function(self, atomic_number, r_cut):
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


# TODO: Find better way of organizing Potential and SlicedPotential
class Potential(object):

    def __init__(self, extent=None, gpts=None, sampling=None, atoms=None, slice_thickness=.5, num_slices=None,
                 corner=None, potential_type='lobato', periodic=True, margin=None, quadrature_samples=25,
                 interp_samples=25, tolerance=1e-2):

        self.grid = Grid(extent=extent, gpts=gpts, sampling=sampling, dimension=2)
        self.slicing = Grid(extent=None, gpts=num_slices, sampling=slice_thickness, dimension=1)

        self.atoms = atoms

        if isinstance(potential_type, str):
            if potential_type.lower() == 'lobato':
                self.potential_type = LobatoPotential()
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()

        if corner is None:
            self.corner = np.zeros(2, dtype=np.float32)

        self.periodic = periodic
        self.quadrature_samples = quadrature_samples
        self.interp_samples = interp_samples
        self.tolerance = tolerance

    @property
    def atoms(self):
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        self._atoms = value

        if value is not None:
            self.slicing.extent = self.atoms.get_cell()[2, 2]

            if self.grid.extent is None:
                self.grid.extent = np.diag(self.atoms.get_cell())[:2]

    def slice(self):
        if self.atoms is None:
            raise RuntimeError()

        functions = {}
        cutoffs = {}
        interp_nodes = {}
        r_min = min(self.grid.sampling)
        for number in np.unique(self.atoms.numbers):
            function = self.potential_type.get_potential_function(number)
            cutoffs[number] = self._find_cutoff(function, self.tolerance)
            functions[number] = self.potential_type.get_soft_potential_function(number, cutoffs[number])
            interp_nodes[number] = self._log_grid(r_min, cutoffs[number], self.interp_samples)

        positions, numbers = self._create_view(self.atoms, self.corner, self.grid.extent, cutoffs)
        quadrature = self._get_quadrature(self.quadrature_samples)
        slice_begin = np.arange(0, self.slicing.extent - self.slicing.sampling, self.slicing.sampling)
        in_slice = self._in_slice(positions, numbers, slice_begin, self.slicing.sampling, cutoffs)
        margin = self._get_margin(self.slicing.extent, self.grid.sampling, cutoffs, self.periodic)

        return SlicedPotential(positions, numbers, functions, cutoffs, interp_nodes, margin, quadrature, self.grid.gpts,
                               self.grid.sampling, slice_begin, self.slicing.sampling, in_slice)

    def _get_margin(self, height, sampling, cutoffs, periodic):
        if periodic:
            return np.ceil(max(cutoffs.values()) / np.min(sampling)).astype(np.int)
        else:
            raise RuntimeError()

    def _in_slice(self, positions, atomic_numbers, slice_begin, slice_thickness, cutoffs):
        cutoffs = np.array([cutoffs[atomic_number] for atomic_number in atomic_numbers])
        return np.abs(slice_begin[:, None] + slice_thickness / 2 - positions[:, 2][None, :]) < (
                cutoffs[None, :] + slice_thickness / 2)

    def _get_quadrature(self, quadrature_samples):
        quadrature = {}
        quadrature['xk'] = np.linspace(-1., 1., quadrature_samples + 1)
        quadrature['wk'] = quadrature['xk'][1:] - quadrature['xk'][:-1]
        quadrature['xk'] = quadrature['xk'][:-1]
        return quadrature

    def _log_grid(self, start, stop, n):
        dt = np.log(stop / start) / (n - 1)
        return (start * np.exp(dt * np.linspace(0., n - 1, n))).astype(np.float32)

    def _find_cutoff(self, function, tolerance):
        return np.float32(brentq(lambda x: function(x) - tolerance, 1e-16, 100))

    def _create_view(self, atoms, corner, extent, cutoffs):
        new_positions = np.zeros((0, 3))
        new_atomic_numbers = np.zeros((0,), dtype=int)
        for atomic_number in np.unique(atoms.get_atomic_numbers()):
            positions = atoms.get_positions()[np.where(atoms.get_atomic_numbers() == atomic_number)[0]]
            positions[:, :2] = (positions[:, :2] - corner) % np.diag(atoms.get_cell())[:2]

            positions = self._add_margin(positions, np.diag(atoms.get_cell())[:2], extent, cutoffs[atomic_number])

            positions = positions[(positions[:, 0] < (extent[0] + cutoffs[atomic_number])) &
                                  (positions[:, 1] < (extent[1] + cutoffs[atomic_number])), :]

            new_positions = np.concatenate((new_positions, positions))
            new_atomic_numbers = np.concatenate((new_atomic_numbers, np.full(len(positions), atomic_number)))

        return new_positions, new_atomic_numbers

    def _add_margin(self, positions, cell, extent, cutoff):
        for axis in [0, 1]:
            nrepeat = np.max((int((extent[axis] + 2 * cutoff) // cell[axis]), 1))
            positions = self._repeat_positions(positions, cell, nrepeat, axis)

            left_positions = positions[positions[:, axis] < (cutoff + extent[axis] - cell[axis] * nrepeat)]
            left_positions[:, axis] += cell[axis] * nrepeat

            right_positions = positions[(nrepeat * cell[axis] - positions[:, axis]) < cutoff]
            right_positions[:, axis] -= nrepeat * cell[axis]

            positions = np.concatenate((positions, left_positions, right_positions), axis=0)

        return positions

    def _repeat_positions(self, positions, cell, n, axis):
        new_positions = positions.copy()
        for i in range(1, n):
            new_positions[:, axis] += cell[axis]
            positions = np.vstack((positions, new_positions))
        return positions


def batch_generator(n_items, max_batch_size):
    n_batches = (n_items + (-n_items % max_batch_size)) // max_batch_size
    batch_size = (n_items + (-n_items % n_batches)) // n_batches

    batch_start = 0
    while 1:
        batch_end = batch_start + batch_size
        if batch_end >= n_items:
            yield batch_start, n_items - batch_end + batch_size
            break
        else:
            yield batch_start, batch_size

        batch_start = batch_end


class SlicedPotential(object):

    def __init__(self, positions, numbers, functions, cutoffs, interp_nodes, margin, quadrature, gpts, sampling,
                 slice_begin, slice_thickness, in_slice):
        self.positions = positions
        self.numbers = numbers
        self.functions = functions
        self.cutoffs = cutoffs
        self.interp_nodes = interp_nodes
        self.margin = margin
        self.quadrature = quadrature
        self.gpts = gpts
        self.sampling = sampling
        self.slice_begin = slice_begin
        self.slice_thickness = slice_thickness
        self.in_slice = in_slice

    def _split_positions(self, indices):
        split_positions = {}
        positions = self.positions[indices]
        numbers = self.numbers[indices]
        for number in np.unique(numbers):
            split_positions[number] = tf.convert_to_tensor(positions[np.where(numbers == number)], tf.float32)
        return split_positions

    def _radials(self, positions, atomic_number, slice_begin, slice_end):
        a = slice_begin - positions[:, 2]
        b = slice_end - positions[:, 2]
        xkab = self.quadrature['xk'][None, :] * ((b - a) / 2)[:, None] + ((a + b) / 2)[:, None]
        wkab = self.quadrature['wk'][None, :] * ((b - a) / 2)[:, None]
        r = tf.sqrt(self.interp_nodes[atomic_number][None, None, :] ** 2 + (xkab ** 2)[:, :, None])
        r = tf.clip_by_value(r, 0, self.interp_nodes[atomic_number][-1])
        return tf.reduce_sum(self.functions[atomic_number](r) * wkab[:, :, None], axis=1)

    @property
    def num_slices(self):
        return self.in_slice.shape[0]

    def slice_generator(self):
        for i in range(self.num_slices):
            yield self._tensor(i)

    def _tensor(self, i, max_atoms=10):
        padded_gpts = self.gpts + 2 * self.margin

        v = tf.Variable(tf.zeros(np.prod(padded_gpts)))

        slice_begin = self.slice_begin[i]
        slice_end = self.slice_begin[i] + self.slice_thickness
        indices = np.where(self.in_slice[i])[0]

        for atomic_number, positions in self._split_positions(indices).items():
            if positions.shape[0] > 0:
                block_margin = tf.cast(self.cutoffs[atomic_number] / min(self.sampling), tf.int32)
                block_size = 2 * block_margin + 1

                x = tf.linspace(0., tf.cast(block_size, tf.float32) * self.sampling[0], block_size)[None, :]
                y = tf.linspace(0., tf.cast(block_size, tf.float32) * self.sampling[1], block_size)[None, :]

                block_indices = tf.range(0, block_size)[None, :] + \
                                tf.range(0, block_size)[:, None] * padded_gpts[1]

                radials = self._radials(positions, atomic_number, slice_begin, slice_end)

                for start, size in batch_generator(positions.shape[0].value, max_atoms):
                    batch_positions = tf.slice(positions, [start, 0], [size, -1])

                    corner_positions = tf.cast(tf.round(batch_positions[:, :2] / self.sampling),
                                               tf.int32) - block_margin + self.margin

                    block_positions = (batch_positions[:, :2] + self.sampling * self.margin
                                       - self.sampling * tf.cast(corner_positions, tf.float32))

                    batch_radials = tf.slice(radials, [start, 0], [size, -1])

                    r_interp = tf.reshape(tf.sqrt(((x - block_positions[:, 0][:, None]) ** 2)[:, :, None] +
                                                  ((y - block_positions[:, 1][:, None]) ** 2)[:, None, :]), (size, -1))

                    r_interp = tf.clip_by_value(r_interp, 0., tf.reduce_max(self.interp_nodes[atomic_number]))

                    v_interp = tf.Variable(tf.reshape(
                        tf.contrib.image.interpolate_spline(
                            tf.tile(self.interp_nodes[atomic_number][None, :], (size, 1))[:, :, None],
                            batch_radials[:, :, None],
                            r_interp[:, :, None], 1), (-1,)))

                    corner_indices = corner_positions[:, 0] * padded_gpts[1] + corner_positions[:, 1]
                    indices = tf.reshape(corner_indices[:, None, None] + block_indices[None, :, :], (-1, 1))

                    tf.scatter_nd_add(v, indices, v_interp)

        v = tf.reshape(v, padded_gpts)
        v = tf.slice(v, (self.margin, self.margin), self.gpts)
        return v  # / utils.kappa


def plane2axes(plane):
    axes = ()
    for axis in list(plane):
        if axis == 'x': axes += (0,)
        if axis == 'y': axes += (1,)
        if axis == 'z': axes += (2,)
    return axes


def display_atoms(source, plane, slice_index=None, scale=100, ax=None, colors=None):
    positions = source.positions
    numbers = source.numbers

    if slice_index is not None:
        positions = positions[source.in_slice[slice_index]]
        numbers = numbers[source.in_slice[slice_index]]

    axes = plane2axes(plane)
    positions = positions[:, axes]
    if colors is None:
        colors = cpk_colors[numbers]
    sizes = covalent_radii[numbers]

    if ax is None:
        fig, ax = plt.subplots()

    box = np.zeros((2, 5))
    if isinstance(source, Atoms):
        box[0, 2:4] += np.diag(source.get_cell())[axes[0]]
        box[1, 1:3] += np.diag(source.get_cell())[axes[1]]
    elif isinstance(source, SlicedPotential):
        extent = source.gpts * source.sampling
        box[0, 2:4] += np.array([extent[0], extent[1], source.slice_begin[-1] + source.slice_thickness])[axes[0]]
        box[1, 1:3] += np.array([extent[0], extent[1], source.slice_begin[-1] + source.slice_thickness])[axes[1]]

    ax.plot(*box, 'k-')

    if isinstance(source, SlicedPotential):
        extent = source.gpts * source.sampling
        box = np.zeros((2, 5))
        for i in range(len(source.in_slice)):
            box[0, :] = np.array([0., 0., source.slice_begin[i]])[axes[0]]
            box[1, :] = np.array([0., 0., source.slice_begin[i]])[axes[1]]
            box[0, 2:4] += np.array([extent[0], extent[1], source.slice_thickness])[axes[0]]
            box[1, 1:3] += np.array([extent[0], extent[1], source.slice_thickness])[axes[1]]
            ax.plot(*box, 'k-', linewidth=1)

    ax.scatter(*positions.T, c=colors, s=scale * sizes)
    ax.axis('equal')
    plt.show()
