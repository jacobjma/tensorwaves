import pytest
import numpy as np
import tensorflow as tf

from ..bases import linspace_no_endpoint, fftfreq, GridProperty, Grid, HasGrid

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


def test_grid():
    grid = Grid(extent=5, sampling=.2)

    assert np.all(grid.extent == np.array([5, 5], dtype=np.float32))
    assert np.all(grid.gpts == np.array([25, 25], dtype=np.int32))
    assert np.all(grid.sampling == np.array([.2, .2], dtype=np.float32))

    grid = Grid(sampling=.2, gpts=10)

    assert np.all(grid.extent == np.array([2, 2], dtype=np.float32))

    grid = Grid(extent=(8, 6), gpts=10)
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
    grid = Grid(gpts=gpts)

    grid.sampling = .1

    assert np.all(grid.extent == np.array([2., 2.], dtype=np.float32))

    grid.sampling = .01
    assert np.all(grid.gpts == np.array([20, 20], dtype=np.int32))
    assert np.all(grid.extent == grid.sampling * np.float32(grid.gpts))

    with pytest.raises(RuntimeError):
        grid.gpts = 10

    grid = Grid()

    with pytest.raises(RuntimeError):
        grid.check_is_defined()


def test_has_grid():
    class Dummy(HasGrid):
        def __init__(self, extent=None, gpts=None, sampling=None, grid=None):
            HasGrid.__init__(self, extent=extent, gpts=gpts, sampling=sampling, grid=grid)

    grid = Grid(extent=5, gpts=10)

    dummy = Dummy(grid=grid)

    assert np.all(dummy.extent == np.array([5, 5], dtype=np.float32))
    assert np.all(dummy.gpts == np.array([10, 10], dtype=np.int32))

    dummy = Dummy(extent=5, gpts=10)

    assert np.all(dummy.sampling == np.array([.5, .5], dtype=np.float32))
