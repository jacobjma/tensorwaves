from itertools import combinations

import numpy as np
import tensorflow as tf
from ase import units

from ..potentials import LobatoPotential, KirklandPotential, log_grid

tf.enable_eager_execution()


def test_similar():
    potentials = (LobatoPotential(), KirklandPotential())

    r = log_grid(.1, 2, 5)

    for potential_a, potential_b in combinations(potentials, 2):
        assert np.all(tf.abs(potential_a._create_function(47)(r) -
                             potential_b._create_function(47)(r)) / potential_b._create_function(47)(r) < .1)


def test_values():
    kappa = 4 * np.pi * units._eps0 / (2 * np.pi * units.Bohr * 1e-10 * units._e)

    potentials = {LobatoPotential(): [10.877785, 3.5969334, 1.1213292, 0.29497656, 0.05587856],
                  KirklandPotential(): [10.966407, 3.7869546, 1.1616056, 0.2839873, 0.04958321]}

    r = np.array([1., 1.316074, 1.7320508, 2.2795072, 3.], dtype=np.float32)

    for potential, values in potentials.items():
        assert np.all(np.isclose(potential._create_function(47)(r) / kappa * 1e20, values))
