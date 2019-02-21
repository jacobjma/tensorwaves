import numpy as np
import pytest
from ase import Atoms

from ..bases import Grid, EnergyProperty
from ..potentials import Potential
from ..waves import WaveFactory


def test_grid():
    grid1 = Grid(extent=10)
    grid2 = Grid()

    grid1.match(grid2)

    assert np.all(grid1.extent == np.array([10, 10], np.float32))
    assert np.all(grid2.extent == np.array([10, 10], np.float32))

    grid2 = Grid(extent=12)

    with pytest.raises(RuntimeError):
        grid1.match(grid2)

    grid2 = Grid(sampling=.1)
    grid1.match(grid2)

    assert np.all(grid1.extent == np.array([10, 10], np.float32))
    assert np.all(grid2.extent == np.array([10, 10], np.float32))
    assert np.all(grid1.extent == np.array([10, 10], np.float32))
    assert np.all(grid2.extent == np.array([10, 10], np.float32))
    assert np.all(grid1.gpts == np.array([100, 100], np.int32))
    assert np.all(grid2.gpts == np.array([100, 100], np.int32))

    grid2 = Grid(sampling=.1, gpts=10)
    with pytest.raises(RuntimeError):
        grid1.match(grid2)


def test_energy():
    energy1 = EnergyProperty(energy=80e3)
    energy2 = EnergyProperty()

    energy1.match(energy2)

    assert energy1.energy == 80.e3
    assert energy2.energy == 80.e3

    energy1 = EnergyProperty()
    energy2 = EnergyProperty(energy=80e3)

    energy1.match(energy2)

    assert energy1.energy == 80.e3
    assert energy2.energy == 80.e3

    energy1.match(energy2)
    energy2 = EnergyProperty(energy=60e3)

    with pytest.raises(RuntimeError):
        energy1.match(energy2)


def test_wavefactory():
    wavefactory = WaveFactory(energy=80e3, sampling=.01)
    atoms = Atoms('C', positions=[(1, 1, 1)], cell=(2, 2, 2))
    potential = Potential(atoms=atoms)

    wavefactory.match(potential)

    print()

    assert np.all(wavefactory.extent == np.array([2, 2], np.float32))
    assert np.all(wavefactory.sampling == np.array([0.01, 0.01], np.float32))
    assert np.all(potential.extent == np.array([2, 2], np.float32))
    assert np.all(potential.sampling == np.array([0.01, 0.01], np.float32))

    wavefactory = WaveFactory(energy=80e3)
    potential = Potential(atoms=atoms, gpts=200)

    wavefactory.match(potential)

    assert np.all(wavefactory.extent == np.array([2, 2], np.float32))
    assert np.all(wavefactory.sampling == np.array([0.01, 0.01], np.float32))
    assert np.all(potential.extent == np.array([2, 2], np.float32))
    assert np.all(potential.sampling == np.array([0.01, 0.01], np.float32))
