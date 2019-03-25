import numpy as np
import tensorflow as tf
from ase import Atoms
from ..potentials import Potential
import pytest

tf.enable_eager_execution()





@pytest.fixture
def potential():
    atoms = Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 4))

    potential = Potential(atoms=atoms, gpts=32, num_slices=5)

    potential.current_slice = 2

    return potential.get_tensor()


def test_potential_centered(potential):
    center = np.where(potential.numpy()[0] == np.max(potential.numpy()[0]))

    assert (center[0] == 16) & (center[1] == 16)


def test_potential_symmetric(potential):
    a = potential.numpy()[0, :, 1:17]
    b = np.fliplr(potential.numpy()[0, :, 16:])

    assert np.all(np.isclose(a, b))

    a = potential.numpy()[0, 1:17]
    b = np.flipud(potential.numpy()[0, 16:])

    assert np.all(np.isclose(a, b))