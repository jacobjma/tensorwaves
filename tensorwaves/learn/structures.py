import itertools

import numpy as np
from ase import Atoms
from scipy.spatial import Voronoi


def wrap_positions(positions, cell, center=(0.5, 0.5), eps=1e-7):
    if not hasattr(center, '__len__'):
        center = (center,) * 2

    shift = np.asarray(center) - 0.5 - eps

    fractional = np.linalg.solve(cell.T, np.asarray(positions).T).T - shift

    for i in range(2):
        fractional[:, i] %= 1.0
        fractional[:, i] += shift[i]

    return np.dot(fractional, cell)


def hexagonal_to_rectangular(sites):
    sites *= (2, 2)
    sites.set_cell([sites.cell[0, 0], sites.cell[1, 1]])
    return sites


def fill_box(sites, box, rotation=0.):
    diagonal = np.hypot(box[0], box[1]) * 2
    n = np.ceil(diagonal / sites.cell[0, 0]).astype(int)
    m = np.ceil(diagonal / sites.cell[1, 1]).astype(int)

    sites *= (n, m)
    sites.set_cell(box)
    sites.rotate(rotation)
    sites.center()
    sites.crop()

    return sites


def is_position_outside(positions, cell):
    fractional = np.linalg.solve(cell.T, np.asarray(positions).T)
    return (fractional[0] < 0) | (fractional[1] < 0) | (fractional[0] > 1) | (fractional[1] > 1)


def repeat_positions(positions, cell, n, m, bothsided=False):
    N = len(positions)

    if bothsided:
        n0, n1 = -n, n + 1
        m0, m1 = -m, m + 1
        new_positions = np.zeros(((2 * n + 1) * (2 * m + 1) * len(positions), 2), dtype=np.float)
    else:
        n0, n1 = 0, n
        m0, m1 = 0, m
        new_positions = np.zeros((n * m * len(positions), 2), dtype=np.float)

    new_positions[:N] = positions.copy()

    k = N
    for i in range(n0, n1):
        for j in range(m0, m1):
            if i + j != 0:
                l = k + N
                new_positions[k:l] = positions + np.dot((i, j), cell)
                k = l

    return new_positions


def voronoi_centroids(points):
    def area_centroid(points):
        points = np.vstack((points, points[0]))
        A = 0
        C = np.zeros(2)
        for i in range(0, len(points) - 1):
            s = points[i, 0] * points[i + 1, 1] - points[i + 1, 0] * points[i, 1]
            A = A + s
            C = C + (points[i, :] + points[i + 1, :]) * s
        return (1 / (3. * A)) * C

    vor = Voronoi(points)

    centroids = np.zeros((len(points), 2))

    j = 0
    for i, point in enumerate(points):
        if all(np.array(vor.regions[vor.point_region[i]]) > -1):
            vertices = vor.vertices[vor.regions[vor.point_region[i]]]
            centroids[j] = area_centroid(vertices)
            j += 1

    return centroids[:j]


def lloyds_relaxation(positions):
    n = len(positions)

    centroids = voronoi_centroids(positions)
    positions = centroids[:n]

    return positions


class SuperCell(object):

    def __init__(self, positions=None, cell=None, arrays=None):

        if positions is None:
            positions = np.zeros((0, 2), dtype=np.float)

        if cell is None:
            self.set_cell(np.zeros((2, 2), dtype=np.float))

        else:
            self.set_cell(np.array(cell, dtype=np.float))

        if arrays is None:
            self._arrays = {}

        else:
            self._arrays = arrays

        self._arrays['positions'] = positions

    @property
    def arrays(self):
        return self._arrays

    @property
    def x(self):
        return self.positions[:, 0]

    @property
    def y(self):
        return self.positions[:, 1]

    @property
    def positions(self):
        return self._arrays['positions']

    @positions.setter
    def positions(self, value):
        self._arrays['positions'] = value

    @property
    def cell(self):
        return self._cell

    @cell.setter
    def cell(self, value):
        self.set_cell(value)

    def set_cell(self, cell):

        cell = np.array(cell, dtype=np.float)

        if cell.shape == (2,):
            cell = np.diag(cell)

        assert cell.shape == (2, 2)

        self._cell = cell

    def __len__(self):
        return len(self.positions)

    def __add__(self, other):
        atoms = self.copy()
        atoms += other
        return atoms

    def __delitem__(self, i):

        if isinstance(i, list) and len(i) > 0:
            # Make sure a list of booleans will work correctly and not be
            # interpreted at 0 and 1 indices.
            i = np.array(i)

        mask = np.ones(len(self), bool)
        mask[i] = False
        for name, a in self.arrays.items():
            self._arrays[name] = a[mask]

    def extend(self, other):
        n1 = len(self)
        n2 = len(other)

        for name, a1 in self.arrays.items():
            a = np.zeros((n1 + n2,) + a1.shape[1:], a1.dtype)
            a[:n1] = a1
            a2 = other.arrays.get(name)
            if a2 is None:
                raise RuntimeError('non-existent array {} in other'.format(name))

        return self

    __iadd__ = extend

    def __imul__(self, m):
        if isinstance(m, int):
            m = (m, m)

        for name, a in self.arrays.items():
            if name != 'positions':
                self._arrays[name] = np.tile(a, (np.product(m),) + (1,) * (len(a.shape) - 1))

        self.arrays['positions'] = repeat_positions(self.positions, self.cell, m[0], m[1])

        self._cell = np.array([m[c] * self.cell[c] for c in range(2)])

        return self

    def repeat(self, rep):
        atoms = self.copy()
        atoms *= rep
        return atoms

    __mul__ = repeat

    def wrap(self):
        self.arrays['positions'][:] = wrap_positions(self.positions, self.cell)

    def center(self, axis=(0, 1), vacuum=None):
        cell = self.cell.copy()
        dirs = np.zeros_like(cell)

        for i in range(2):
            dirs[i] = np.array([-cell[i - 1][1], cell[i - 1][0]])
            dirs[i] /= np.sqrt(np.dot(dirs[i], dirs[i]))

            if np.dot(dirs[i], cell[i]) < 0.0:
                dirs[i] *= -1

        cell = self.cell.copy()

        if isinstance(axis, int):
            axes = (axis,)
        else:
            axes = axis

        p = self.arrays['positions']

        longer = np.zeros(2)
        shift = np.zeros(2)
        for i in axes:
            p0 = np.dot(p, dirs[i]).min() if len(p) else 0
            p1 = np.dot(p, dirs[i]).max() if len(p) else 0
            height = np.dot(cell[i], dirs[i])

            if vacuum is not None:
                lng = (p1 - p0 + 2 * vacuum) - height
            else:
                lng = 0.0

            top = lng + height - p1
            shf = 0.5 * (top - p0)
            cosphi = np.dot(cell[i], dirs[i]) / np.sqrt(np.dot(cell[i], cell[i]))

            longer[i] = lng / cosphi
            shift[i] = shf / cosphi

        translation = np.zeros(2)
        for i in axes:
            nowlen = np.sqrt(np.dot(cell[i], cell[i]))
            if vacuum is not None or self.cell[i].any():
                self.cell[i] = cell[i] * (1 + longer[i] / nowlen)
                translation += shift[i] * cell[i] / nowlen

        self.arrays['positions'] += translation

    def relax(self, n=1):

        positions = self.positions

        N = len(positions)
        for i in range(n):
            positions = repeat_positions(positions, self.cell, 1, 1, True)

            # import matplotlib.pyplot as plt
            # print(positions)

            centroids = voronoi_centroids(positions)
            positions = centroids[:N]

            positions = wrap_positions(positions, self.cell)

            # print(centroids)

            # plt.plot(*centroids.T)
            # plt.show()
            # sss

        self.arrays['positions'][:] = positions

    def crop(self):
        del self[is_position_outside(self.positions, self.cell)]

    def rotate(self, angle, center=None):

        if center is None:
            center = np.sum(self.cell, axis=1) / 2

        angle = angle / 180. * np.pi

        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.arrays['positions'][:] = np.dot(R, self.positions.T - np.array(center)[:, None]).T

    def copy(self):
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        positions = arrays['positions']
        del arrays['positions']
        return self.__class__(positions=positions, cell=self._cell.copy(), arrays=arrays)


class Label(object):

    def __init__(self, positions=None, structure_classes=None, cell=None):
        if structure_classes is None:
            structure_classes = np.zeros(len(positions), dtype=np.int)

        assert len(positions) == len(structure_classes)

        self._positions = np.array(positions)
        arrays = {'structure_classes': np.array(structure_classes)}

        super().__init__(positions=positions, arrays=arrays, cell=cell)

    @property
    def structure_classes(self):
        return self._arrays['structure_classes']

    def copy(self):
        positions = self.arrays['positions'].copy()
        structure_classes = self.arrays['structure_classes'].copy()
        return self.__class__(positions=positions, structure_classes=structure_classes, cell=self.cell.copy())


class Site(object):

    def __init__(self, structures, probabilities=None, flip=False):

        if probabilities is not None:
            assert len(structures) == len(probabilities)
            self._probabilities = np.array(probabilities) / np.sum(probabilities)

        elif len(structures) == 1:
            self._probabilities = None

        else:
            self._probabilities = np.full(len(structures), 1. / len(structures))

        self._structures = structures

        self._flip = flip

    def choose(self):
        i = np.random.choice(np.arange(len(self._structures)), 1, p=self._probabilities)[0]

        structure = self._structures[i]

        if self._flip:
            if np.random.rand() < .5:
                positions = structure.positions
                positions[:, 2] = -positions[:, 2]
                structure.positions = positions

        return structure


class Sites(SuperCell):

    def __init__(self, sites=None, positions=None, cell=None):
        if sites is None:
            sites = []

        arrays = {'sites': np.array(sites)}

        super().__init__(positions=positions, cell=cell, arrays=arrays)

    @property
    def sites(self):
        return self.arrays['sites']

    def as_atoms(self):

        cell = np.zeros((3, 3))

        cell[:2, :2] = self.cell

        combined_atoms = Atoms(cell=cell)
        for site, position in zip(self.sites, self.positions):
            atoms = site.choose().copy()
            atoms.positions[:, :2] += position
            combined_atoms += atoms

        combined_atoms.center(axis=2, vacuum=2)

        return combined_atoms
