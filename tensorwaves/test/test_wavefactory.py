from ..waves import PlaneWaves
import numpy as np
import pytest


def test_planewaves():
    waves = PlaneWaves(extent=10, gpts=50, energy=80e3)

    assert np.all(waves.extent == np.array([10, 10], dtype=np.float32))
    assert np.all(waves.gpts == np.array([50, 50], dtype=np.int32))
    assert np.all(waves.sampling == np.array([.2, .2], dtype=np.float32))

    tensorwaves = waves.get_tensor()

    assert np.all(tensorwaves.extent == np.array([10, 10], dtype=np.float32))
    assert np.all(tensorwaves.gpts == np.array([50, 50], dtype=np.int32))
    assert np.all(tensorwaves.sampling == np.array([.2, .2], dtype=np.float32))

    assert waves.up_to_date == True
    assert tensorwaves == waves.get_tensor()

    waves.clear_tensor()
    assert tensorwaves != waves.get_tensor()

    assert waves.up_to_date == True

    waves.gpts = 100

    assert waves.up_to_date == False

    tensorwaves = waves.get_tensor()

    assert np.all(tensorwaves.extent == np.array([10, 10], dtype=np.float32))
    assert np.all(tensorwaves.gpts == np.array([100, 100], dtype=np.int32))
    assert np.all(tensorwaves.sampling == np.array([.1, .1], dtype=np.float32))

    waves = PlaneWaves(extent=10, energy=80e3)

    with pytest.raises(RuntimeError):
        waves.get_tensor()

    waves = PlaneWaves(extent=10, gpts=50)
    with pytest.raises(RuntimeError):
        waves.get_tensor()
