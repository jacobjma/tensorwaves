import numpy as np
from ase import Atoms

from ..potentials import Potential
from ..waves import PlaneWaves


def test_tem():
    atoms = Atoms('CSiCuAuU', positions=[(x, 25, 5) for x in np.linspace(5, 45, 5)], cell=(50, 50, 10))

    potential = Potential(atoms=atoms, gpts=512, parametrization='kirkland')

    wave = PlaneWaves(energy=200e3)

    wave = wave.multislice(potential)

    wave = wave.apply_ctf(defocus=700, Cs=-1.3e7, aperture_radius=.01037)

    image = wave.detect()

    assert np.allclose(image.numpy()[0, ::32, 256], [1.0071477, 1.0019431, 0.9894879, 1.0007411, 1.0109925, 0.9310855,
                                                     0.99967396, 0.9988709, 0.89847034, 1.0050664, 1.002859, 0.8524643,
                                                     1.016926, 1.0130365, 0.9333292, 1.0250149])
