import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from ase import Atoms

from tensorwaves import utils
from tensorwaves.display import Display
import traitlets
import traittypes
import bqplot


class AtomsView(traitlets.HasTraits):
    cutoffs = traitlets.Dict()
    start = traittypes.Array()
    size = traittypes.Array()
    atoms = traitlets.Instance(Atoms)

    positions = traittypes.Array(read_only=True)
    atomic_numbers = traittypes.Array(read_only=True)
    origin = traittypes.Array(read_only=True)

    slices = traitlets.Int(allow_none=True, default_value=None)
    slice_thickness = traitlets.Float(allow_none=True, default_value=None)
    in_slice = traittypes.Array(read_only=True)

    @property
    def box(self):
        return np.hstack((self.size, [self.cell[2]]))

    @property
    def cell(self):
        return np.diag(self.atoms.get_cell())

    @traitlets.default('origin')
    def _default_origin(self):
        return np.array([0., 0., 0.])

    @traitlets.default('start')
    def _default_start(self):
        return np.array([0., 0.])

    @traitlets.default('size')
    def _default_size(self):
        return np.diag(self.atoms.get_cell())[:2]

    @traitlets.observe('atoms')
    def _observe_atoms(self, change):
        self._new_view()

    def _new_view(self):
        new_positions = np.zeros((0, 3))
        new_atomic_numbers = np.zeros((0,), dtype=int)
        for atomic_number in np.unique(self.atoms.get_atomic_numbers()):
            positions = self.atoms.get_positions()[np.where(self.atoms.get_atomic_numbers() == atomic_number)[0]]
            positions[:, :2] = (positions[:, :2] - self.start) % self.cell[:2]

            positions = self._add_margin(positions, self.cell, self.size, self.cutoffs[atomic_number])

            positions = positions[(positions[:, 0] < (self.size[0] + self.cutoffs[atomic_number])) &
                                  (positions[:, 1] < (self.size[1] + self.cutoffs[atomic_number])), :]

            new_positions = np.concatenate((new_positions, positions))
            new_atomic_numbers = np.concatenate((new_atomic_numbers, np.full(len(positions), atomic_number)))

        self.set_trait('positions', new_positions)
        self.set_trait('atomic_numbers', new_atomic_numbers)

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

    @traitlets.observe('slice_thickness')
    def _observe_slice_thickness(self, change):
        self.slices = int(np.ceil(self.box[2] / change['new']))

    @traitlets.observe('slices')
    def _observe_slices(self, change):
        self.slice_thickness = self.box[2] / change['new']
        self._slice()

    def _slice(self):
        slice_centers = np.linspace(self.slice_thickness / 2, self.box[2] - self.slice_thickness / 2, self.slices)
        cutoffs = np.array([self.cutoffs[atomic_number] for atomic_number in self.atomic_numbers])
        self.set_trait('in_slice', np.abs(slice_centers[:, None] - self.positions[:, 2][None, :]) < (
                cutoffs[None, :] + self.slice_thickness / 2))

    def slice_indices(self, i, atomic_number=None):
        if atomic_number is None:
            return np.where(self.in_slice[i])[0]
        else:
            return np.where(self.in_slice[i] & (self.atomic_numbers == atomic_number))[0]

    def __getitem__(self, i):
        if i < -self.slices or i >= self.slices:
            raise IndexError('slice index out of range')

        return AtomsSlice(view=self, index=i)

    def __iter__(self):
        for i in range(self.slices):
            yield self[i]

    def display(self, plane='xy'):
        return ProjectedAtomsDisplay(view=self, plane=plane).show()


class AtomsSlice(traitlets.HasTraits):
    view = traitlets.Instance(AtomsView)
    index = traitlets.Int()

    @property
    def thickness(self):
        return self.view.slice_thickness

    @property
    def origin(self):
        return np.array([0., 0., self.index * self.thickness])

    @property
    def box(self):
        return np.array([self.view.box[0], self.view.box[1], self.thickness])

    @property
    def positions(self):
        return self.view.positions[self.view.slice_indices(self.index)]

    @property
    def atomic_numbers(self):
        return self.view.atomic_numbers[self.view.slice_indices(self.index)]

    def display(self, plane='xy'):
        return ProjectedAtomsDisplay(view=self, plane=plane).show()


class ProjectedAtomsDisplay(Display):
    view = traitlets.Union([traitlets.Instance(Po), traitlets.Instance(AtomsSlice)])
    plane = traitlets.Union([traitlets.Tuple(default_value=(0, 1)), traitlets.Unicode()])

    @traitlets.validate('plane')
    def _validate_plane(self, proposal):
        if isinstance(proposal['value'], str):
            value = ()
            for axis in list(proposal['value']):
                if axis == 'x': value += (0,)
                if axis == 'y': value += (1,)
                if axis == 'z': value += (2,)
            return value
        else:
            return proposal['value']

    def _get_positions(self):
        return self.view.positions[:, self.plane].T

    def _get_colors(self):
        colors = cpk_colors[self.view.atomic_numbers]
        return list(np.apply_along_axis(lambda x: "#{:02x}{:02x}{:02x}".format(*x), 1, (colors * 255).astype(np.int)))

    def _get_atomic_sizes(self):
        return covalent_radii[self.view.atomic_numbers]

    def _get_box(self):
        box = np.zeros((2, 5))
        box[0, :] = self.view.origin[self.plane[0]]
        box[1, :] = self.view.origin[self.plane[1]]
        box[0, 2:4] += self.view.box[self.plane[0]]
        box[1, 1:3] += self.view.box[self.plane[1]]
        return box

    @traitlets.default('marks')
    def _default_marks(self):
        x, y = self._get_positions()
        atoms = bqplot.Scatter(x=x, y=y, scales={'x': self.x_scale, 'y': self.y_scale},
                               colors=self._get_colors(), stroke_width=1.5, stroke='black')

        x, y = self._get_box()
        box = bqplot.Lines(x=x, y=y, scales={'x': self.x_scale, 'y': self.y_scale}, colors=['black'])
        return [atoms, box]
