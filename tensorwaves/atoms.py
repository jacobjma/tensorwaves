import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import cpk_colors

from tensorwaves import utils


def axis2idx(axis):
    if axis == 'x':
        return 0
    if axis == 'y':
        return 1
    if axis == 'z':
        return 2


def display_atoms(atoms, plane='xy', ax=None, scale=1, linewidth=2, edgecolor='k', **kwargs):
    if ax is None:
        ax = plt.subplot()

    axes_idx = [axis2idx(axis) for axis in list(plane)]

    positions = atoms.positions[:, axes_idx]
    atomic_numbers = atoms.atomic_numbers

    c = cpk_colors[atomic_numbers]
    s = scale * covalent_radii[atomic_numbers]

    scatter = ax.scatter(*positions.T, c=c, s=s, linewidth=linewidth, edgecolor=edgecolor, **kwargs)
    ax.axis('equal')

    rect_x = np.full(5, atoms.origin[axes_idx[0]])
    rect_x[2:4] += atoms.box[axes_idx[0]]
    rect_y = np.full(5, atoms.origin[axes_idx[1]])
    rect_y[1:3] += atoms.box[axes_idx[1]]
    ax.plot(rect_x, rect_y, 'k', linewidth=1.5)

    if axes_idx[1] == 2:
        ax.invert_yaxis()

    return ax, scatter


class AtomsView(object):

    def __init__(self, atoms):

        self.set_atoms(atoms)

    atoms = property(lambda self: self._atoms)
    positions = property(lambda self: self._positions)
    atomic_numbers = property(lambda self: self._atomic_numbers)
    box = property(lambda self: self._box)
    origin = property(lambda self: np.array([0., 0., 0.]))
    cutoffs = property(lambda self: self._cutoffs)
    n_slices = property(lambda self: self._n_slices)
    slice_thickness = property(lambda self: self.box[2] / self.n_slices)

    def set_atoms(self, atoms):

        if not utils.cell_is_rectangular(atoms.get_cell()):
            raise RuntimeError()

        self._atoms = atoms
        self._positions = None
        self._atomic_numbers = None
        self._box = None
        self._cutoffs = None
        self._n_slices = None
        self._in_slice = None

    def create_view(self, cutoffs, begin=None, size=None):
        if begin is None:
            begin = np.array([0., 0.])

        cell = np.diag(self._atoms.get_cell()).copy()

        if size is None:
            size = cell[:2]

        new_positions = np.zeros((0, 3))
        new_atomic_numbers = np.zeros((0,), dtype=int)
        for atomic_number in np.unique(self._atoms.get_atomic_numbers()):
            positions = self._atoms.get_positions()[np.where(self._atoms.get_atomic_numbers() == atomic_number)[0]]
            positions[:, :2] = (positions[:, :2] - begin) % cell[:2]

            positions = self._add_margin(positions, cell, size, cutoffs[atomic_number])

            positions = positions[(positions[:, 0] < (size[0] + cutoffs[atomic_number])) &
                                  (positions[:, 1] < (size[1] + cutoffs[atomic_number])), :]

            new_positions = np.concatenate((new_positions, positions))
            new_atomic_numbers = np.concatenate((new_atomic_numbers, np.full(len(positions), atomic_number)))

        self._cutoffs = cutoffs
        self._positions = new_positions
        self._atomic_numbers = new_atomic_numbers
        self._box = np.array([size[0], size[1], cell[2]])

    def _add_margin(self, positions, cell, size, cutoff):
        for axis in [0, 1]:
            nrepeat = np.max((int((size[axis] + 2 * cutoff) // cell[axis]), 1))
            positions = self._repeat_positions(positions, cell, nrepeat, axis)

            left_positions = positions[positions[:, axis] < (cutoff + size[axis] - cell[axis] * nrepeat)]
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

    def slice(self, n_slices=None, slice_thickness=None):
        if (n_slices is None) & (slice_thickness is not None):
            self._n_slices = np.int(np.ceil(self.box[2] / slice_thickness))
        elif n_slices is not None:
            self._n_slices = n_slices
        else:
            raise RuntimeError()

        slice_centers = np.linspace(self.slice_thickness / 2, self.box[2] - self.slice_thickness / 2, self.n_slices)

        cutoffs = np.array([self.cutoffs[atomic_number] for atomic_number in self.atomic_numbers])
        self._in_slice = (np.abs(slice_centers[:, None] - self.positions[:, 2][None, :]) < (
                cutoffs[None, :] + self.slice_thickness / 2))

    def slice_indices(self, i, atomic_number=None):
        if atomic_number is None:
            return np.where(self._in_slice[i])[0]
        else:
            return np.where(self._in_slice[i] & (self._atomic_numbers == atomic_number))[0]

    def __getitem__(self, i):
        if i < -self.n_slices or i >= self.n_slices:
            raise IndexError('slice index out of range')

        return AtomsSlice(self, i)

    def __iter__(self):
        for i in range(self.n_slices):
            yield self[i]

    def display(self, plane='xy', ax=None, scale=1):
        return display_atoms(self, plane=plane, ax=ax, scale=scale)


class AtomsSlice(object):

    def __init__(self, atoms_view, index):
        self._index = index
        self._atoms_view = atoms_view

    index = property(lambda self: self._index)
    atoms_view = property(lambda self: self._atoms_view)

    @property
    def thickness(self):
        return self.atoms_view.slice_thickness

    @property
    def origin(self):
        return np.array([0., 0., self.index * self.thickness])

    @property
    def box(self):
        return np.array([self.atoms_view.box[0], self.atoms_view.box[1], self.thickness])

    @property
    def positions(self):
        return self.atoms_view.positions[self.atoms_view.slice_indices(self.index)]

    @property
    def atomic_numbers(self):
        return self.atoms_view.atomic_numbers[self.atoms_view.slice_indices(self.index)]

    def display(self, plane='xy', ax=None, scale=1):
        return display_atoms(self, plane=plane, ax=ax, scale=scale)
