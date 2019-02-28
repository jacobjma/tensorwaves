import numpy as np

from ..waves import PrismWaves
from ..zernike import ZernikeExpansion
from ..transfer import polar2zernike, zernike2polar, ZernikeAberrations

import tensorflow as tf
import pytest

tf.enable_eager_execution()


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

    polar['C41'] = np.random.uniform(-1, 1) * 1e4
    polar['phi41'] = np.random.uniform(-np.pi, np.pi)
    polar['C43'] = np.random.uniform(-1, 1) * 1e4
    polar['phi43'] = np.random.uniform(-np.pi, np.pi)
    polar['C45'] = np.random.uniform(-1, 1) * 1e4
    polar['phi45'] = np.random.uniform(-np.pi, np.pi)

    polar['C50'] = np.random.uniform(-1, 1) * 1e5
    polar['C52'] = np.random.uniform(-1, 1) * 1e5
    polar['phi52'] = np.random.uniform(-np.pi, np.pi)
    polar['C54'] = np.random.uniform(-1, 1) * 1e5
    polar['phi54'] = np.random.uniform(-np.pi, np.pi)
    polar['C56'] = np.random.uniform(-1, 1) * 1e5
    polar['phi56'] = np.random.uniform(-np.pi, np.pi)

    return polar


def test_zernike(polar):
    prism = PrismWaves(extent=10, sampling=.1, energy=80e3, cutoff=.035)
    S = prism.get_tensor()
    S.aberrations.set_parameters(polar)

    z = S.aberrations.parametrization.to_zernike(aperture_radius=.035)

    x, y = S.alpha_x.numpy(), S.alpha_y.numpy()
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(x, y)

    assert np.allclose(np.exp(-2 * np.pi * 1.j * z(r, phi).numpy() / S.wavelength), S.get_coefficients().numpy())


def test_convert(polar):
    zernike1 = polar2zernike(polar, .035)
    zernike2 = polar2zernike(zernike2polar(zernike1, .035), .035)

    for (key1, value1), (key2, value2) in zip(zernike1.items(), zernike2.items()):
        assert np.isclose(value1, value2)
