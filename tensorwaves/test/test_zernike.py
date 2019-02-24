import numpy as np

from ..waves import PrismWaves
from ..zernike import polar2zernike, ZernikeExpansion

import tensorflow as tf

tf.enable_eager_execution()


def test_zernike():
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

    prism = PrismWaves(extent=10, sampling=.1, energy=80e3, cutoff=.035)
    S = prism.get_tensor()
    S.aberrations.parameters.set_parameters(polar)

    zernike = polar2zernike(polar, aperture_radius=.035)

    expansion = ZernikeExpansion(S.alpha_x.numpy() / .035, S.alpha_y.numpy() / .035, zernike)

    assert np.allclose(np.exp(-2 * np.pi * 1.j * expansion.sum() / S.wavelength), S.get_coefficients().numpy())
