import csv
import math

import numpy as np
import tensorflow as tf
from ase import Atoms
from ase.data import chemical_symbols
from scipy.optimize import brentq

from tensorwaves import utils, plotutils
#from tensorwaves.bases import FactoryBase, TensorBase
from tensorwaves.parametrization import potentials


def find_cutoff(func, min_value, xmin=1e-16, xmax=5):
    return np.float32(brentq(lambda x: func(x) - min_value, xmin, xmax))


def load_parameter_file(filename):
    parameters = {}
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        keys = next(reader)
        for _, row in enumerate(reader):
            values = list(map(float, row))
            parameters[int(row[0])] = dict(zip(keys, values))

    return parameters


def tanh_sinh_quadrature(m, h):
    xk = tf.Variable(tf.zeros(2 * m))
    wk = tf.Variable(tf.zeros(2 * m))
    for i in range(0, 2 * m):
        k = i - m
        xk[i].assign(tf.tanh(math.pi / 2 * tf.sinh(k * h)))
        numerator = h / 2 * math.pi * tf.cosh(k * h)
        denominator = tf.cosh(math.pi / 2 * tf.sinh(k * h)) ** 2
        wk[i].assign(numerator / denominator)
    return xk, wk


def riemann_quadrature(m):
    xk = tf.lin_space(-1., 1., m + 1)
    wk = xk[1:] - xk[:-1]
    xk = xk[:-1]
    return xk, wk


class ParameterizedPotential(object):

    def __init__(self, parametrization='lobato', parameters=None):

        self.parametrization = parametrization

        if parameters is None:
            self.parameters = load_parameter_file(potentials[parametrization]['default_parameters'])
        elif isinstance(str, parameters):
            self.parameters = load_parameter_file(parameters)
        else:
            self.parameters = parameters

    def get_analytic_projection(self, atomic_number):
        return potentials[self.parametrization]['projected_potential'](self.parameters[atomic_number])

    def get_potential(self, atomic_number):
        return potentials[self.parametrization]['potential'](self.parameters[atomic_number])

    def get_soft_potential(self, atomic_number, r_cut):
        return potentials[self.parametrization]['soft_potential'](self.parameters[atomic_number], r_cut)

    def get_cutoff(self, atomic_number, tolerance):
        v = self.get_potential(atomic_number)
        return find_cutoff(v, tolerance, xmin=1e-3, xmax=10)


class Potential(FactoryBase, ParameterizedPotential):

    def __init__(self, atoms=None, nslices=None, gpts=None, sampling=None, parametrization='lobato', parameters=None):

        if not utils.cell_is_rectangular(atoms.get_cell()):
            raise RuntimeError()

        self.atoms = atoms

        self._nslices = nslices

        FactoryBase.__init__(self, gpts=gpts, extent=self.extent, sampling=sampling)
        ParameterizedPotential.__init__(self, parametrization=parametrization, parameters=parameters)

    def slice(self, nslices, tolerance=1e-3, m=20, n=20):
        return SlicedPotential(self, nslices, tolerance=tolerance, m=m, n=n)

    @property
    def box(self):
        if self.atoms is None:
            return None
        return np.diag(self.atoms.get_cell()).astype(np.float32)

    @property
    def extent(self):
        if self.atoms is None:
            return None
        return np.diag(self.atoms.get_cell()[:2]).astype(np.float32)

    @property
    def entrance_plane(self):
        return 0.

    @property
    def exit_plane(self):
        return (self.atoms.get_cell()[2, 2]).astype(np.float32)

    @extent.setter
    def extent(self, _):
        raise RuntimeError()


class SlicedPotential(FactoryBase):

    def __init__(self, potential, nslices, tolerance=1e-2, m=20, n=20):

        super().__init__(gpts=potential.gpts, extent=potential.extent, sampling=potential.sampling)

        self._positions = potential.atoms.get_positions()
        self._atomic_numbers = potential.atoms.get_atomic_numbers()
        self._exit_plane = np.float32(potential.atoms.get_cell()[2, 2])

        self._cutoffs = {}
        self._functions = {}
        self._log_grids = {}
        for atomic_number in np.unique(self.atomic_numbers):
            self._cutoffs[atomic_number] = potential.get_cutoff(atomic_number, tolerance)
            self._functions[atomic_number] = potential.get_soft_potential(atomic_number, self._cutoffs[atomic_number])
            self._log_grids[atomic_number] = utils.log_grid(min(self.sampling),
                                                            self._cutoffs[atomic_number] * np.float32(2), n)

        self._set_margin(max(self._cutoffs.values()))



        self._set_slices(nslices, cutoffs)

        self._corner_positions = ((self.positions[:, :2] - cutoffs[:, None]) / self.sampling).astype(int)
        self._block_positions = self.positions[:, :2] - self._corner_positions * self.sampling
        self._set_quadrature(m)

    @property
    def nslices(self):
        return self._nslices

    @property
    def positions(self):
        return self._positions

    @property
    def atomic_numbers(self):
        return self._atomic_numbers

    @property
    def exit_plane(self):
        return self._exit_plane

    @property
    def padded_gpts(self):
        max_blocksize = int(2 * max(self._cutoffs.values()) / min(self.sampling) + 1)
        return np.array((self.gpts[0] + max_blocksize, self.gpts[1] + max_blocksize))



    def _set_quadrature(self, m):
        self._xk, self._wk = riemann_quadrature(m)

    def _set_slices(self, nslices, cutoffs):
        self._nslices = nslices
        self._slice_thickness = self.exit_plane / nslices

        slice_centers = np.linspace(self._slice_thickness / 2, self.exit_plane - self._slice_thickness / 2, nslices)

        self._in_slice = (np.abs(slice_centers[:, None] - self.positions[:, 2][None, :]) < cutoffs[None, :])

    def _set_margin(self, margin):

        self._margin = margin

        lattice_vectors = np.zeros((2, 3), dtype=np.float32)
        np.fill_diagonal(lattice_vectors, self.extent)

        for i in range(2):
            mask = self.positions[:, i] < margin
            left_positions = self.positions[mask] + lattice_vectors[i]
            left_numbers = self.atomic_numbers[mask]

            mask = (self.extent[i] - self.positions[:, i]) < margin
            right_positions = self.positions[mask] - lattice_vectors[i]
            right_numbers = self.atomic_numbers[mask]

            self._positions = np.concatenate([self.positions, left_positions, right_positions], axis=0)
            self._atomic_numbers = np.concatenate([self.atomic_numbers, left_numbers, right_numbers], axis=0)

    @property
    def slice_thickness(self):
        return self._slice_thickness

    def slice_natoms(self, i, atomic_number):
        return len(self.slice_indices(i, atomic_number))

    def slice_entrance_plane(self, i):
        return i * self.slice_thickness

    def slice_exit_plane(self, i):
        return self.slice_entrance_plane(i) + self.slice_thickness

    def slice_positions(self, i, atomic_number=None):
        return self.positions[self.slice_indices(i, atomic_number)]

    def blocksize(self, atomic_number):
        return np.float32(2 * self._cutoffs[atomic_number] / min(self.sampling) + 1)

    def radial_potential(self, i, positions, atomic_number):
        a = self.slice_entrance_plane(i) - positions[:, 2]
        b = self.slice_exit_plane(i) - positions[:, 2]
        xkab = self._xk[None, :] * ((b - a) / 2)[:, None] + ((a + b) / 2)[:, None]
        wkab = self._wk[None, :] * ((b - a) / 2)[:, None]
        r_radial = tf.sqrt(self._log_grids[atomic_number][None, None, :] ** 2 + (xkab ** 2)[:, :, None])
        r_radial = tf.clip_by_value(r_radial, 0, self._cutoffs[atomic_number])
        v_radial = tf.reduce_sum(self._functions[atomic_number](r_radial) * wkab[:, :, None], axis=1)

        return v_radial

    def slice_tensor(self, i, max_atoms):
        v = tf.Variable(tf.zeros(np.prod(self.padded_gpts)))

        for atomic_number in np.unique(self.atomic_numbers):
            if self.slice_natoms(i, atomic_number) > 0:
                positions = np.float32(self.slice_positions(i, atomic_number))

                pixel_margin = np.int32(self._cutoffs[atomic_number] / min(self.sampling))
                blocksize = 2 * pixel_margin + 1

                x = tf.linspace(0., blocksize * self.sampling[0], blocksize)[None, :]
                y = tf.linspace(0., blocksize * self.sampling[1], blocksize)[None, :]

                block_indices = tf.range(0, blocksize)[None, :] + tf.range(0, blocksize)[:, None] * self.padded_gpts[1]

                for start, size in utils.batch_generator(len(positions), max_atoms):
                    batch_positions = tf.slice(positions, [start, 0], [size, -1])
                    corner_positions = tf.cast(tf.round(batch_positions[:, :2] / self.sampling),
                                               tf.int32) - pixel_margin
                    block_positions = batch_positions[:, :2] - tf.cast(corner_positions, tf.float32) * self.sampling

                    r_radial = tf.tile(self._log_grids[atomic_number][None, :], (size, 1))

                    v_radial = self.radial_potential(i, batch_positions, atomic_number)

                    r_interp = tf.reshape(tf.sqrt(((x - block_positions[:, 0][:, None]) ** 2)[:, :, None] +
                                                  ((y - block_positions[:, 1][:, None]) ** 2)[:, None, :]), (size, -1))

                    v_interp = tf.Variable(tf.reshape(
                        tf.contrib.image.interpolate_spline(r_radial[:, :, None], v_radial[:, :, None],
                                                            r_interp[:, :, None], 1), (-1,)))

                    corner_indices = corner_positions[:, 0] * self.padded_gpts[1] + corner_positions[:, 1]

                    indices = tf.reshape(corner_indices[:, None, None] + block_indices[None, :, :], (-1, 1))

                    tf.scatter_nd_add(v, indices, v_interp)

        v = tf.reshape(v, self.padded_gpts)
        v = tf.slice(v, (0, 0), self.gpts)
        return v / utils.kappa

    def __getitem__(self, i):
        return PotentialSlice(self, i)


class PotentialSlice(object):

    def __init__(self, sliced_potential, index):
        self._sliced_potential = sliced_potential
        self.index = index

    @property
    def sliced_potential(self):
        return self._sliced_potential

    @property
    def indices(self):
        return self.sliced_potential.slice_indices(self.index)

    @property
    def positions(self):
        return self.sliced_potential._positions[self.indices]

    @property
    def atomic_numbers(self):
        return self.sliced_potential._atomic_numbers[self.indices]

    @property
    def entrance_plane(self):
        return self.index * self.thickness

    @property
    def exit_plane(self):
        return self.entrance_plane + self.thickness

    @property
    def thickness(self):
        return self.sliced_potential._slice_thickness

    @property
    def box(self):
        return np.hstack((self.sliced_potential.extent, self.thickness))

    @property
    def extent(self):
        return self.sliced_potential.extent

    def tensor(self, max_atoms=10):
        return self.sliced_potential.slice_tensor(self.index, max_atoms)

    def show_atoms(self, **kwargs):
        plotutils.show_atoms(self, **kwargs)

    def show(self, **kwargs):
        image = self.tensor().numpy()
        extent = [0, self.extent[0], 0, self.extent[1]]
        plotutils.show_image(image, extent=extent, **kwargs)
