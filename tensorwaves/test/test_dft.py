import numpy as np
import pytest
from ase import Atoms

from ..potentials import Potential


@pytest.mark.slow
def test_dft():
    from gpaw import GPAW, PW
    from gpaw.utilities.ps2ae import PS2AE
    atoms = Atoms('C', positions=[(2, 2, 2)], cell=(4, 4, 4), pbc=True)
    calc = GPAW(mode=PW(600), eigensolver='cg', gpts=(40, 40, 40), txt=None)
    atoms.set_calculator(calc)
    atoms.get_potential_energy()

    ps2ae = PS2AE(atoms.calc, h=.025)
    potential_array = -ps2ae.get_electrostatic_potential(ae=True)
    potential_array -= np.min(potential_array)
    potential_array = np.sum(potential_array, axis=2) / 40

    potential = Potential(atoms=atoms, gpts=160, num_slices=1, num_nodes=300, tolerance=1e-4)
    potential.current_slice = 0

    difference = potential.get_tensor().numpy()[0] - potential_array
    relative_difference = difference[40:-40, 40:-40] / (potential_array[40:-40, 40:-40] + 1e-6)
    relative_difference = np.abs(relative_difference) < .1

    assert np.sum(relative_difference == 0) <= 1
