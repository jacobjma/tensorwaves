import numpy as np
import tensorflow as tf
import pytest
from tensorwaves.transfer import PhaseAberration, polar2cartesian, cartesian2polar


@pytest.fixture
def polar():
    polar = {}
    polar['C10'] = np.random.uniform(-1, 1)
    polar['C12'] = np.random.uniform(-1, 1)
    polar['phi12'] = np.random.uniform(-np.pi, np.pi)

    polar['C21'] = np.random.uniform(-1, 1) * 1e2
    polar['phi21'] = np.random.uniform(-np.pi, np.pi)
    polar['C23'] = np.random.uniform(-1, 1) * 1e2
    polar['phi23'] = np.random.uniform(-np.pi, np.pi)

    polar['C30'] = np.random.uniform(-1, 1) * 1e3
    polar['C32'] = np.random.uniform(-1, 1) * 1e3
    polar['phi32'] = np.random.uniform(-np.pi, np.pi)
    polar['C34'] = np.random.uniform(-1, 1) * 1e3
    polar['phi34'] = np.random.uniform(-np.pi, np.pi)
    return polar


def test_cartesian_aberrations(polar):
    cartesian = polar2cartesian(polar)

    cartesian_aberrations = PhaseAberration(extent=(10, 5), gpts=56, energy=80e3, parametrization='cartesian',
                                            **cartesian)

    polar_aberrations = PhaseAberration(extent=(10, 5), gpts=56, energy=80e3, parametrization='polar', **polar)

    assert np.all(np.abs(cartesian_aberrations.get_tensor().numpy()[0] -
                         polar_aberrations.get_tensor().numpy()[0]) < 5e-4)


def test_conversion(polar):
    cartesian1 = polar2cartesian(polar)
    cartesian2 = polar2cartesian(cartesian2polar(cartesian1))

    for (key1, value1), (key2, value2) in zip(cartesian1.items(), cartesian2.items()):
        assert np.isclose(value1, value2)
