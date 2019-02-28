import numpy as np
from tensorwaves.waves import ProbeWaves
import tensorflow as tf

tf.enable_eager_execution()


def test_probe():
    uncorrected = ProbeWaves(extent=16, gpts=512, energy=100e3, aperture_radius=0.0103)
    uncorrected.positions = (8, 8)
    uncorrected.aberrations.parametrization.defocus = 674
    uncorrected.aberrations.parametrization.Cs = -1.3e7

    corrected = ProbeWaves(extent=16, gpts=512, energy=100e3, aperture_radius=0.025)
    corrected.positions = (8, 8)

    uncorrected_image = np.abs(uncorrected.get_tensor().numpy()) ** 2
    uncorrected_image /= np.sum(uncorrected_image)

    corrected_image = np.abs(corrected.get_tensor().numpy()) ** 2
    corrected_image /= np.sum(corrected_image)

    assert np.isclose(uncorrected_image[0, 256, 256] / corrected_image[0, 256, 256], 0.13939089)
