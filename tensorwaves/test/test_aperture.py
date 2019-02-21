import tensorflow as tf

from ..transfer import Aperture

tf.enable_eager_execution()


def test_aperture():

    aperture = Aperture(extent=10, gpts=100, energy=1)

