import numpy as np
import tensorflow as tf

from ..potentials import KirklandPotential, log_grid, PotentialProjector

tf.enable_eager_execution()


def test_potential_projection():
    projector = PotentialProjector(num_samples=800)

    potential = KirklandPotential()
    projected_function = potential._create_projected_function(47)
    function = potential._create_function(47)

    r = log_grid(.5, 2, 50)

    assert np.all(np.isclose(projector.project(function, r, -5, 5)[0], projected_function(r), atol=0.1))

