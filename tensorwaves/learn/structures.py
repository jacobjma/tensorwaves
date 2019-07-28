import numpy as np
from ase import Atoms

from scipy.spatial import Voronoi, ConvexHull
from matplotlib.path import Path
from tensorwaves.learn.augment import bandpass_noise_2d


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


def random_graphene(box, a=2.46, border=.5):
    basis = [(0, 0), (2 / 3., 1 / 3.)]
    cell = [[a, 0], [-a / 2., a * 3 ** 0.5 / 2.]]
    positions = np.dot(np.array(basis), np.array(cell))

    supercell = SuperCell(positions=positions, cell=cell)

    supercell = hexagonal_to_rectangular(supercell)

    supercell = fill_box(supercell, box, np.random.rand() * 360, border=border)

    supercell.relax(1)

    return supercell


def pseudo_graphene(box):
    n = int(2 / (2.46 * 4.261) * box[0] * box[1])

    cell = np.diag(box)

    positions = np.array([[np.random.uniform(0, l) for l in box[:2]] for i in range(n)])

    positions = lloyds_relaxation(positions, 50, cell)

    positions = repeat_positions(positions, cell, 1, 1, True)
    vor = Voronoi(positions)
    vertices = vor.vertices

    vertices = vertices[vertices[:, 0] > 0.]
    vertices = vertices[vertices[:, 1] > 0.]
    vertices = vertices[vertices[:, 0] < box[0]]
    vertices = vertices[vertices[:, 1] < box[1]]

    positions = lloyds_relaxation(vertices, 1, cell)

    return SuperCell(positions=positions, cell=cell)


# def insert_in_hole(supercell_1, supercell_2):
#     polygon = blob() * np.random.uniform(10, 40) + 5
#     path = Path(polygon)
#
#     inside = path.contains_points(supercell_1.positions)
#
#     positions_1 = positions_1[inside == False]
#
#     positions_2 =
#
#     return self + other


def fill_box(supercell, box, rotation=0., border=0.):
    diagonal = np.hypot(box[0], box[1]) * 2
    n = np.ceil(diagonal / supercell.cell[0, 0]).astype(int)
    m = np.ceil(diagonal / supercell.cell[1, 1]).astype(int)

    supercell *= (n, m)
    supercell.set_cell(np.array(box) - border)
    supercell.rotate(rotation)
    supercell.center()
    supercell.crop()
    supercell.set_cell(box)
    supercell.positions += border / 2

    return supercell


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


def lloyds_relaxation_iter(positions):
    n = len(positions)

    centroids = voronoi_centroids(positions)
    positions = centroids[:n]

    return positions


def lloyds_relaxation(positions, n, cell):
    cell = np.array(cell)
    if cell.shape == (2,):
        cell = np.diag(cell)

    N = len(positions)
    for i in range(n):
        positions = repeat_positions(positions, cell, 1, 1, True)

        centroids = voronoi_centroids(positions)
        positions = centroids[:N]

        positions = wrap_positions(positions, cell)

    return positions


def interpolate_smoothly(points, N):
    interpolated = np.fft.fft(points)
    half = (len(points) + 1) // 2
    interpolated = np.concatenate((interpolated[:half],
                                   np.zeros(len(points) * N),
                                   interpolated[half:]))
    return [x.real / len(points) for x in np.fft.fft(interpolated)[::-1]]


def blob(n=5, m=5):
    points = np.random.rand(n, 2)
    points = points[ConvexHull(points).vertices]
    points = np.array([interpolate_smoothly(point, m) for point in zip(*points)]).T
    return np.array(points)


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

            a[n1:] = a2

            self._arrays[name] = a

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
        positions = lloyds_relaxation(positions, n, self.cell)
        self.arrays['positions'][:] = positions

    def crop(self):
        del self[is_position_outside(self.positions, self.cell)]

    def rotate(self, angle, center=None):

        if center is None:
            center = np.sum(self.cell, axis=1) / 2

        angle = angle / 180. * np.pi

        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        self.arrays['positions'][:] = np.dot(R, self.positions.T - np.array(center)[:, None]).T + center

    def make_hole(self, lower_corner, size):

        polygon = lower_corner + blob() * size
        path = Path(polygon)

        inside = path.contains_points(self.positions)

        del self[inside]

    def random_strain(self, scale, amplitude, direction):

        def lookup_nearest(x0, y0, x, y, z):
            xi = np.abs(x - x0).argmin()
            yi = np.abs(y - y0).argmin()
            return z[yi, xi]

        noise = bandpass_noise_2d(inner=0, outer=scale, shape=(32, 32))
        x = np.linspace(0, self.cell[0, 0], 32)
        y = np.linspace(0, self.cell[1, 1], 32)

        self.positions[:, direction] += amplitude * np.array(
            [lookup_nearest(p[0], p[1], x, y, noise) for p in self.positions]).T

    def copy(self):
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        positions = arrays['positions']
        del arrays['positions']
        return self.__class__(positions=positions, cell=self._cell.copy(), arrays=arrays)


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

        self.flip = flip

    @property
    def structures(self):
        return self._structures

    @property
    def probabilities(self):
        return self._probabilities

    def choose(self):
        i = np.random.choice(np.arange(len(self._structures)), 1, p=self._probabilities)[0]

        structure = self._structures[i]

        if self.flip:
            if np.random.rand() < .5:
                positions = structure.positions
                positions[:, 2] = structure.cell[2, 2] - positions[:, 2]
                structure.positions = positions

        return structure

    def copy(self):
        structures = [structure.copy() for structure in self.structures]
        return self.__class__(structures, self.probabilities.copy(), self.flip)


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

    def copy(self):
        sites = [site.copy() for site in self.sites]
        return self.__class__(sites=sites, positions=self.positions.copy(), cell=self.cell.copy())
