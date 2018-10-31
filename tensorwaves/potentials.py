import csv
import math

import numpy as np
import tensorflow as tf
from ase import Atoms
from ase.data import chemical_symbols
from scipy.optimize import brentq

from tensorwaves import utils
from tensorwaves.atoms import SlicedAtoms
from tensorwaves.bases import FactoryBase, TensorBase
from tensorwaves.parametrization import potentials
from tensorwaves.atoms import AtomsSlice


def find_cutoff(func, min_value, xmin=1e-16, xmax=5):
    return brentq(lambda x: func(x) - min_value, xmin, xmax)


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


def radial_grids(x, y, margin_range, positions, pixel_positions):
    return tf.reshape(tf.sqrt(((tf.reshape(tf.batch_gather(x, tf.clip_by_value(tf.reshape(pixel_positions[:, 0][:, None]
                                                                                          + margin_range[None, :],
                                                                                          [-1]), 0, x.shape[0])),
                                           (positions.shape[0], margin_range.shape[0])) -
                                positions[:, 0][:, None]) ** 2)[:, :, None] +

                              ((tf.reshape(tf.batch_gather(y, tf.clip_by_value(tf.reshape(pixel_positions[:, 1][:, None]
                                                                                          + margin_range[None, :],
                                                                                          [-1]), 0, y.shape[0])),
                                           (positions.shape[0], margin_range.shape[0])) -
                                positions[:, 1][:, None]) ** 2)[:, None, :]),
                      (positions.shape[0], -1))


def scatter_sum(reference, updates, pixel_positions, margin):
    for i in range(updates.shape[0]):
        xa = pixel_positions[i, 0] - margin
        xb = pixel_positions[i, 0] + margin + 1
        ya = pixel_positions[i, 1] - margin
        yb = pixel_positions[i, 1] + margin + 1

        xa_clip = tf.math.maximum(xa, 0)
        xb_clip = tf.math.minimum(xb, reference.shape[0])
        ya_clip = tf.math.maximum(ya, 0)
        yb_clip = tf.math.minimum(yb, reference.shape[1])

        if (xa_clip < xb_clip) & (ya_clip < yb_clip):
            update = tf.reshape(updates[i], (2 * margin + 1, 2 * margin + 1))
            update = tf.slice(update, [xa_clip - xa, ya_clip - ya], [update.shape[0].value +
                                                                     xb_clip - xb - xa_clip + xa,
                                                                     update.shape[1].value +
                                                                     yb_clip - yb - ya_clip + ya])

            reference[xa_clip:xb_clip, ya_clip:yb_clip].assign(reference[xa_clip:xb_clip, ya_clip:yb_clip] + update)


class Potential(FactoryBase):

    def __init__(self, atoms=None, gpts=None, sampling=None, parametrization='lobato', parameters=None):

        self.parametrization = parametrization

        if parameters is None:
            self.parameters = load_parameter_file(potentials[parametrization]['default_parameters'])
        elif isinstance(str, parameters):
            self.parameters = load_parameter_file(parameters)
        else:
            self.parameters = parameters

        self.set_atoms(atoms)

        super().__init__(gpts=gpts, extent=self.extent, sampling=sampling)

    def __len__(self):
        return len(self._sliced_atoms)

    def check_buildable(self):
        if (self.gpts is None) | (self.sliced_atoms is None):
            raise RuntimeError('the atoms or gpts are not defined')

    def sliced_atoms(self):
        return self._sliced_atoms

    def set_atoms(self, atoms):
        if isinstance(atoms, Atoms):
            self._sliced_atoms = SlicedAtoms(atoms)
        elif isinstance(atoms, SlicedAtoms):
            self._sliced_atoms = atoms

    def get_analytic_projection(self, atomic_number):
        return potentials[self.parametrization]['projected_potential'](self.parameters[atomic_number])

    def get_potential(self, atomic_number):
        return potentials[self.parametrization]['potential'](self.parameters[atomic_number])

    def get_soft_potential(self, atomic_number, r_cut):
        return potentials[self.parametrization]['soft_potential'](self.parameters[atomic_number], r_cut)

    def get_cutoff(self, atomic_number, tolerance):
        v = self.get_potential(atomic_number)
        return find_cutoff(v, tolerance, xmin=1e-3, xmax=10)

    def get_radial_grid(self, atomic_number, n, rmin=None):
        if rmin is None:
            rmin = min(self.sampling)
        return utils.log_grid(rmin, self._cutoffs[atomic_number] * 2, n)

    def slice(self, nslices, tolerance=1e-3, m=20, n=20):

        self._sliced_atoms.set_n_slices(nslices)

        self._x, self._y = self.linspace()
        self._xk, self._wk = riemann_quadrature(m)
        # self._v = tf.Variable(tf.zeros(self.gpts))

        self._cutoffs = {}
        self._functions = {}
        self._log_grids = {}
        self._margins = {}
        for atomic_number in self._sliced_atoms._unique_atomic_numbers:
            self._cutoffs[atomic_number] = self.get_cutoff(atomic_number, tolerance)
            self._functions[atomic_number] = self.get_soft_potential(atomic_number, self._cutoffs[atomic_number])
            self._log_grids[atomic_number] = self.get_radial_grid(atomic_number, n)
            self._margins[atomic_number] = tf.cast(self._cutoffs[atomic_number] / min(self.sampling), tf.int32)

        self._sliced_atoms.add_margin(max(self._cutoffs.values()))

    @property
    def extent(self):
        if self._sliced_atoms is None:
            return None
        return self._sliced_atoms.extent

    @extent.setter
    def extent(self, _):
        raise RuntimeError()

    def __getitem__(self, i):
        return PotentialSlice(self, self._sliced_atoms, i)


class PotentialSlice(AtomsSlice):

    def __init__(self, potential, sliced_atoms, slice_index):
        self._potential = potential
        super().__init__(sliced_atoms, slice_index)

    def function(self, atomic_number):
        return self.potential._functions[atomic_number]

    def margin(self, atomic_number):
        return self.potential._margins[atomic_number]

    def log_grid(self, atomic_number):
        return self.potential._log_grids[atomic_number]

    def cutoff(self, atomic_number):
        return self.potential._cutoffs[atomic_number]

    @property
    def tensor(self):
        return self._potential.tensor

    @property
    def potential(self):
        return self._potential

    @property
    def sampling(self):
        return self.potential.sampling

    @property
    def gpts(self):
        return self.potential.gpts

    @property
    def xk(self):
        return self.potential._xk

    @property
    def wk(self):
        return self.potential._wk

    @property
    def x(self):
        return self.potential._x

    @property
    def y(self):
        return self.potential._y

    @property
    def pixelized_positions(self):
        return tf.cast(tf.round(self.positions[:, :2] / self._potential.sampling), tf.int32)

    def calculate(self, max_atoms=80):
        v = tf.Variable(tf.zeros(self.gpts))

        entrance_plane = self.entrance_plane
        exit_plane = self.exit_plane

        all_positions = self.positions
        all_pixel_positions = tf.cast(tf.round(all_positions[:, :2] / self.sampling), tf.int32)
        atomic_numbers = self.atomic_numbers

        for atomic_number in self.unique_atomic_numbers:
            margin_range = tf.range(-self.margin(atomic_number), self.margin(atomic_number) + 1)

            function = self.function(atomic_number)
            positions = tf.boolean_mask(all_positions, tf.math.equal(atomic_numbers, atomic_number))
            pixel_positions = tf.boolean_mask(all_pixel_positions, tf.math.equal(atomic_numbers, atomic_number))

            for batch_start, batch_size in batch_generator(positions.shape[0].value, max_atoms):
                batch_positions = tf.slice(positions, [batch_start, 0], [batch_size, -1])
                batch_pixel_positions = tf.slice(pixel_positions, [batch_start, 0], [batch_size, -1])

                a = entrance_plane - batch_positions[:, 2]
                b = exit_plane - batch_positions[:, 2]
                xkab = self.xk[None, :] * ((b - a) / 2)[:, None] + ((a + b) / 2)[:, None]
                wkab = self.wk[None, :] * ((b - a) / 2)[:, None]

                r = tf.clip_by_value(
                    tf.sqrt(self.log_grid(atomic_number)[None, None, :] ** 2 + (xkab ** 2)[:, :, None]), 0,
                    self.cutoff(atomic_number))

                vr = tf.reduce_sum(function(r) * wkab[:, :, None], axis=1)

                r = radial_grids(self.x, self.y, margin_range, batch_positions, batch_pixel_positions)[:, :, None]
                # r = tf.clip_by_value(r, 0, self.cutoff(atomic_number))

                # print(function(self.cutoff(atomic_number)))

                v_proj = tf.contrib.image.interpolate_spline(
                    tf.tile(self.log_grid(atomic_number)[None, :], (batch_size, 1))[:, :, None],
                    vr[:, :, None], r, 1)
                # v_proj = r

                # v_proj = tf.where(r < self.cutoff(atomic_number), r, tf.zeros(r.shape))
                # v_proj = function(r)
                scatter_sum(v, v_proj, batch_pixel_positions, self.margin(atomic_number))

        return v


class TransmissionTensor(object):

    def __init__(self, tensor, thickness, energy, extent=None, sampling=None):
        super().__init__(tensor=tensor, energy=energy, extent=extent, sampling=sampling, dimension=2)

    def slice_generator(self):
        pass
