import itertools

import numpy as np
from ase import Atoms
from scipy.spatial import Voronoi

from tensorwaves.learn.augment import bandpass_noise_2d


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

    centroids = []

    for i, point in enumerate(points):
        if all(np.array(vor.regions[vor.point_region[i]]) > -1):
            vertices = vor.vertices[vor.regions[vor.point_region[i]]]
            centroids.append(area_centroid(vertices))

    return np.array(centroids)


def repeat_positions(positions, box):
    new_positions = positions.copy()

    for direction in set(itertools.combinations([-1, 0, 1] * 2, 2)):
        if direction != (0, 0):
            new_positions = np.concatenate((new_positions, positions + direction * np.array(box)))

    return new_positions


def lloyds_relaxation(atoms, num_iter=1, bc='periodic', wrap=True):
    def mirror(positions, box):
        new_positions = positions.copy()

        for i, j in zip([0, 0, 1, 1], [2, 0, 2, 0]):
            tmp_positions = positions.copy()
            tmp_positions[:, i] = j * np.array(box)[i] - tmp_positions[:, i]
            new_positions = np.concatenate((new_positions, tmp_positions))

        return new_positions

    positions = atoms.get_positions()[:, :2]
    box = np.diag(atoms.get_cell())[:2]

    N = len(positions)

    for i in range(num_iter):
        if box is not None:
            if bc == 'periodic':
                positions = repeat_positions(positions, box)
            elif bc == 'mirror':
                positions = mirror(positions, box)
            else:
                raise NotImplementedError('Boundary condition {0} not recognized'.format(bc))

        centroids = voronoi_centroids(positions)
        positions = centroids[:N]

        if wrap:
            positions[:, 0] = positions[:, 0] % box[0]
            positions[:, 1] = positions[:, 1] % box[1]

    atoms.set_positions(np.hstack((positions, atoms.get_positions()[:, 2, None])))
    return atoms


def hexagonal2orthogonal(atoms):
    atoms = atoms.repeat((1, 2, 1))
    cell = atoms.get_cell()
    cell[1, 0] = 0
    atoms.set_cell(cell)
    atoms.wrap()
    return atoms


def fill_box_2d(atoms, box, rotation):
    diagonal = np.hypot(box[0], box[1]) * 2
    n = np.ceil(diagonal / atoms.get_cell()[0, 0]).astype(int)
    m = np.ceil(diagonal / atoms.get_cell()[1, 1]).astype(int)

    atoms *= (n, m, 1)

    atoms.set_cell(box)
    atoms.rotate(rotation, 'z')
    atoms.center()

    del atoms[atoms.get_positions()[:, 0] < 0]
    del atoms[atoms.get_positions()[:, 0] > box[0]]
    del atoms[atoms.get_positions()[:, 1] < 0]
    del atoms[atoms.get_positions()[:, 1] > box[1]]
    return atoms


def grains(box, centers, sheet_funcs, p=1):
    def mirror(points, box):
        original = points.copy()

        for i, j in zip([0, 0, 1, 1], [2, 0, 2, 0]):
            new_points = original.copy()
            new_points[:, i] = j * np.array(box)[i] - new_points[:, i]
            points = np.concatenate((points, new_points))

        return points

    def in_hull(p, hull):
        from scipy.spatial import Delaunay
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0

    centers = mirror(centers, box)

    vor = Voronoi(centers)
    atoms = Atoms()
    for i, center in enumerate(centers):

        if (center[0] > 0) & (center[1] > 0) & (center[0] < box[0]) & (center[1] < box[1]):
            region = vor.vertices[vor.regions[vor.point_region[i]]]

            sheet_func = np.random.choice(sheet_funcs, p=p)
            new_atoms = sheet_func(box)

            new_atoms = new_atoms[in_hull(new_atoms.get_positions()[:, :2], region)]
            atoms += new_atoms

    atoms.set_cell(box)
    atoms.set_pbc(1)
    atoms.wrap()
    return atoms


def random_strain(atoms, direction, amplitude, scale, gpts=(32, 32)):
    positions = atoms.get_positions()
    cell = np.diag(atoms.get_cell())

    def lookup_nearest(x0, y0, x, y, z):
        xi = np.abs(x - x0).argmin()
        yi = np.abs(y - y0).argmin()
        return z[yi, xi]

    noise = bandpass_noise_2d(inner=0, outer=scale, shape=gpts)
    x = np.linspace(0, cell[0], gpts[0])
    y = np.linspace(0, cell[1], gpts[1])

    positions[:, direction] += amplitude * np.array([lookup_nearest(p[0], p[1], x, y, noise) for p in positions]).T
    atoms.set_positions(positions)
    return atoms
