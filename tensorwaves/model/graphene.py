import numpy as np
from ase.lattice.hexagonal import Graphite

from tensorwaves.noise import power_law_noise


def sheet(box, rotation, edge_tol=1):
    atoms = Graphite(symbol='C', latticeconstant={'a': 2.46, 'c': 6.70},
                     directions=[[1, -2, 1, 0], [2, 0, -2, 0], [0, 0, 0, 1]], size=(1, 1, 1))

    del atoms[atoms.get_positions()[:, 2] < 2]

    diagonal = np.hypot(box[0], box[1]) * 2
    n = np.ceil(diagonal / atoms.get_cell()[0, 0]).astype(int)
    m = np.ceil(diagonal / atoms.get_cell()[1, 1]).astype(int)

    atoms *= (n, m, 1)

    atoms.set_cell(box)
    atoms.rotate(rotation, 'z')
    atoms.center()

    del atoms[atoms.get_positions()[:, 0] < edge_tol]
    del atoms[atoms.get_positions()[:, 0] > box[0] - edge_tol]
    del atoms[atoms.get_positions()[:, 1] < edge_tol]
    del atoms[atoms.get_positions()[:, 1] > box[1] - edge_tol]

    return atoms


def apply_random_strain(atoms, direction, power=-3, amplitude=10 ** 3, gpts=(32, 32)):
    positions = atoms.get_positions()
    cell = np.diag(atoms.get_cell())

    def lookup_nearest(x0, y0, x, y, z):
        xi = np.abs(x - x0).argmin()
        yi = np.abs(y - y0).argmin()
        return z[yi, xi]

    noise = power_law_noise(gpts=gpts, sampling=(1., 1.), power=power)
    x = np.linspace(0, cell[0], gpts[0])
    y = np.linspace(0, cell[1], gpts[1])

    positions[:, direction] += amplitude * np.array([lookup_nearest(p[0], p[1], x, y, noise) for p in positions]).T

    atoms.set_positions(positions)

    return atoms
