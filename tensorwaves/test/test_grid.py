import pytest
import numpy as np
import tensorflow as tf

from ..bases import linspace_no_endpoint, fftfreq, GridProperty, Grid, Tensor, TensorWithEnergy
from ..transfer import FrequencyTransfer, Aperture, TemporalEnvelope, PhaseAberration, CTF
from ..potentials import Potential
from ..waves import WaveFactory, PlaneWaves, ProbeWaves, PrismWaves
from .test_utils import CallCounter

tf.enable_eager_execution()


def test_linspace_no_endpoint():
    assert np.all(np.isclose(linspace_no_endpoint(0., 9, 8), np.linspace(0., 9, 8, endpoint=False, dtype=np.float32)))


def test_fftfreq():
    assert np.all(np.isclose(fftfreq(9, .3), np.fft.fftfreq(9, .3).astype(np.float32)))
    assert np.all(np.isclose(fftfreq(10, .3), np.fft.fftfreq(10, .3).astype(np.float32)))


def test_grid_property():
    grid_property = GridProperty(value=5, dtype=np.float32)

    assert np.all(grid_property.value == np.array([5, 5], dtype=np.float32))
    assert grid_property.value.dtype == np.float32

    grid_property.value = 2

    assert np.all(grid_property.value == np.array([2, 2], dtype=np.float32))
    assert grid_property.value.dtype == np.float32

    grid_property = GridProperty(value=[5, 5], dtype=np.int32)

    assert np.all(grid_property.value == np.array([5, 5], dtype=np.int32))
    assert grid_property.value.dtype == np.int32

    grid_property = GridProperty(value=lambda: np.array([5, 5]), dtype=np.float32, locked=True)

    assert np.all(grid_property.value == np.array([5, 5], dtype=np.float32))
    assert grid_property.value.dtype == np.float32

    with pytest.raises(RuntimeError):
        grid_property.value = 2

    with pytest.raises(RuntimeError):
        GridProperty(value=[5, 5, 5], dtype=np.float32)

    grid_property = GridProperty(value=None, dtype=np.float32)

    grid_property.value = 2

    assert np.all(grid_property.value == np.array([2, 2], dtype=np.float32))


@pytest.mark.parametrize('has_grid',
                         [Grid,  # Tensor, TensorWithEnergy,
                          FrequencyTransfer, Aperture, TemporalEnvelope, PhaseAberration, CTF,
                          WaveFactory, PlaneWaves, ProbeWaves, PrismWaves])
def test_has_grid(has_grid):
    grid = has_grid(extent=5, sampling=.2)

    assert np.all(grid.extent == np.array([5, 5], dtype=np.float32))
    assert np.all(grid.gpts == np.array([25, 25], dtype=np.int32))
    assert np.all(grid.sampling == np.array([.2, .2], dtype=np.float32))

    grid = has_grid(sampling=.2, gpts=10)

    assert np.all(grid.extent == np.array([2, 2], dtype=np.float32))

    grid = has_grid(extent=(8, 6), gpts=10)
    assert np.all(grid.sampling == np.array([0.8, 0.6], dtype=np.float32))

    grid.sampling = .2
    assert np.all(grid.extent == np.array([8, 6], dtype=np.float32))
    assert np.all(grid.gpts == np.array([40, 30], dtype=np.int32))

    grid.gpts = 100
    assert np.all(grid.extent == np.array([8, 6], dtype=np.float32))
    assert np.all(grid.sampling == np.array([0.08, 0.06], dtype=np.float32))

    grid.extent = (16, 12)
    assert np.all(grid.sampling == np.array([0.08, 0.06], dtype=np.float32))
    assert np.all(grid.gpts == np.array([200, 200], dtype=np.int32))

    grid.extent = (10, 10)
    assert np.all(grid.sampling == grid.extent / np.float32(grid.gpts))

    grid.sampling = .3
    assert np.all(grid.extent == grid.sampling * np.float32(grid.gpts))

    grid.gpts = 30
    assert np.all(grid.sampling == grid.extent / np.float32(grid.gpts))

    gpts = GridProperty(value=lambda: np.array([20, 20], dtype=np.int32), dtype=np.int32, locked=True)
    grid = has_grid(gpts=gpts)

    grid.sampling = .1

    assert np.all(grid.extent == np.array([2., 2.], dtype=np.float32))

    grid.sampling = .01
    assert np.all(grid.gpts == np.array([20, 20], dtype=np.int32))
    assert np.all(grid.extent == grid.sampling * np.float32(grid.gpts))

    with pytest.raises(RuntimeError):
        grid.gpts = 10

    grid = has_grid()

    with pytest.raises(RuntimeError):
        grid.check_is_defined()


@pytest.mark.parametrize('tensorfactory_with_grid',
                         [FrequencyTransfer, Aperture, TemporalEnvelope, PhaseAberration, CTF,
                          WaveFactory, PlaneWaves, ProbeWaves, PrismWaves])
def test_grid_update(tensorfactory_with_grid):
    instance = tensorfactory_with_grid(extent=10, sampling=.1)

    counter = CallCounter(lambda: instance.extent)
    instance._calculate_tensor = counter.func_caller
    instance.check_is_defined = lambda: None

    assert instance.up_to_date == False
    assert np.all(instance.get_tensor() == np.array([10, 10], dtype=np.float32))
    assert instance.up_to_date == True
    assert counter.n == 1
    instance.get_tensor()
    assert counter.n == 1

    instance.extent = 5
    assert instance.up_to_date == False
    assert np.all(instance.get_tensor() == np.array([5, 5], dtype=np.float32))
    assert instance.up_to_date == True
    assert counter.n == 2

    instance.extent = 5
    assert instance.up_to_date == True
    instance.get_tensor()
    assert counter.n == 2

    #instance = tensorfactory_with_grid()
    #with pytest.raises(RuntimeError) as exc_info:
    #    instance.get_tensor()
    #
    #assert str(exc_info.value) == 'grid is not defined'
