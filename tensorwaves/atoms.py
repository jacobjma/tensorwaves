import tensorflow as tf

from tensorwaves.bases import Box, ZAxis
from ase import data as ase_data
from tensorwaves import utils


class SlicedAtoms(Box):

    def __init__(self, atoms, nslices=None, margin=None):
        cell = atoms.get_cell()
        extent = tf.constant([cell[0, 0], cell[1, 1]], tf.float32)
        exit_plane = tf.cast(cell[2, 2], tf.float32)

        if not utils.cell_is_rectangular(atoms.get_cell()):
            raise RuntimeError()

        super().__init__(exit_plane=exit_plane, extent=extent)

        self._atomic_numbers = tf.constant(atoms.get_atomic_numbers(), tf.int32)
        self._positions = tf.constant(atoms.get_positions(), tf.float32)
        self._unique_atomic_numbers = list(tf.unique(self.atomic_numbers)[0].numpy())

        if nslices is not None:
            self.set_n_slices(nslices)
        else:
            self._slice_thickness = None
            self._nslices = None

        if margin is not None:
            self.add_margin(margin)
        else:
            self._margin = margin

    def __len__(self):
        return self._nslices

    @property
    def unique_atomic_numbers(self):
        return self._unique_atomic_numbers

    @property
    def chemical_symbols(self):
        return [ase_data.chemical_symbols[number] for number in self._atomic_numbers]

    @property
    def atomic_numbers(self):
        return self._atomic_numbers

    @property
    def positions(self):
        return self._positions

    def _check_slice_valid(self, i):
        if i > self._nslices - 1:
            raise RuntimeError()

    def set_n_slices(self, nslices):
        self._slice_thickness = self.exit_plane / nslices
        self._nslices = nslices

    def slice_limits(self, i):
        self._check_slice_valid(i)
        entrance_plane = i * self._slice_thickness
        return entrance_plane, entrance_plane + self._slice_thickness

    def slice_indices(self, i):
        self._check_slice_valid(i)
        a, b = self.slice_limits(i)
        return tf.where((self.positions[:, 2] > (a - self._margin)) & (self.positions[:, 2] < (b + self._margin)))

    def remove_margin(self):
        self._positions = self._positions[:self._natoms]
        self._atomic_numbers = self._atomic_numbers[:self._natoms]
        self._margin = None

    def add_margin(self, margin):
        if margin == self._margin:
            return

        if self._margin is not None:
            self.remove_margin()
        self._margin = margin
        self._natoms = self._positions.shape[0].value

        lattice_vectors = tf.zeros((2, 3), dtype=tf.float32)
        lattice_vectors = tf.linalg.set_diag(lattice_vectors, self.extent)

        for i in range(2):
            mask = self._positions[:, i] < self._margin
            left_positions = tf.boolean_mask(self._positions, mask) + lattice_vectors[i]
            left_numbers = tf.boolean_mask(self._atomic_numbers, mask)

            mask = (self.extent[i] - self._positions[:, i]) < self._margin
            right_positions = tf.boolean_mask(self._positions, mask) - lattice_vectors[i]
            right_numbers = tf.boolean_mask(self._atomic_numbers, mask)

            self._positions = tf.concat([self._positions, left_positions, right_positions], axis=0)
            self._atomic_numbers = tf.concat([self._atomic_numbers, left_numbers, right_numbers], axis=0)

    def __getitem__(self, i):
        return AtomsSlice(self, i)


class AtomsSlice(ZAxis):

    def __init__(self, sliced_atoms, slice_index):
        self._sliced_atoms = sliced_atoms
        self._slice_index = slice_index
        self._indices = self.sliced_atoms.slice_indices(slice_index)
        entrance_plane, exit_plane = self.sliced_atoms.slice_limits(slice_index)

        super().__init__(entrance_plane=entrance_plane, exit_plane=exit_plane)

    @property
    def sliced_atoms(self):
        return self._sliced_atoms

    @property
    def box(self):
        return tf.concat((self._sliced_atoms.extent, [self.depth]), 0)

    @property
    def positions(self):
        return tf.gather_nd(self._sliced_atoms.positions, self._indices)

    @property
    def atomic_numbers(self):
        return tf.gather_nd(self._sliced_atoms.atomic_numbers, self._indices)

    @property
    def unique_atomic_numbers(self):
        return list(tf.unique(self.atomic_numbers)[0].numpy())

    def __len__(self):
        return self._indices.shape[0].value
