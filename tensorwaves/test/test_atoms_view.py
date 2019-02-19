import numpy as np
import pytest
import tensorflow as tf
from ase import Atoms
from mock import Mock

from ..potentials import Potential, PotentialParameterization

tf.enable_eager_execution()


@pytest.fixture
def mock_parameterization():
    return Mock(spec=PotentialParameterization)


def test_atoms_view(mock_parameterization):
    atoms = Atoms('C', positions=[(1, 1, 1)], cell=(2, 2, 2))

    potential = Potential(atoms=atoms, slice_thickness=0.5, parametrization=mock_parameterization)

    assert np.all(np.isclose(potential.extent, np.diag(atoms.get_cell())[:2]))
    assert np.all(np.isclose(potential.origin, [0., 0.]))
    assert potential.thickness == atoms.get_cell()[2, 2]
    assert potential.num_slices == np.int(np.ceil(atoms.get_cell()[2, 2] / 0.5))
    assert np.isclose(potential.slice_thickness, atoms.get_cell()[2, 2] / potential.num_slices)
    assert potential.slice_entrance == 0.
    assert np.isclose(potential.slice_exit - potential.slice_entrance, potential.slice_thickness)

    potential.current_slice = potential.num_slices - 1

    assert np.isclose(potential.slice_exit, potential.thickness)

    with pytest.raises(RuntimeError):
        potential.current_slice = potential.num_slices

    mock_parameterization.get_cutoff.return_value = 0.

    assert len(potential.get_positions(6)) == 1

    mock_parameterization.get_cutoff.return_value = 2

    assert len(potential.get_positions(6)) == 1

    potential.new_view()

    assert len(potential.get_positions(6)) == 9

    potential.new_view(origin=(.5, .5))

    assert len(potential.get_positions(6)) == 9

    potential.new_view(origin=(.5, .5), extent=(4, 4))

    assert len(potential.get_positions(6)) == 16
