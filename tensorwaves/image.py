import numpy as np

from tensorwaves.bases import Tensor


class Image(Tensor):

    def __init__(self, tensor, extent=None, sampling=None, space='direct'):
        Tensor.__init__(self, tensor, extent=extent, sampling=sampling, space=space)

    # def get_normalization_factor(self):
    #     return tf.reduce_sum(self._tensor) * tf.reduce_prod(self.grid.sampling) * tf.cast(
    #         tf.reduce_prod(self.grid.gpts), tf.float32)
    #
    # def normalize(self, dose):
    #     self._tensor = self._tensor * dose / self.get_normalization_factor()

    # def scale_intensity(self, scale):
    #     self._tensor = self._tensor * scale
    #
    # def apply_poisson_noise(self):
    #     self._tensor = tf.random.poisson(self._tensor, shape=[1])[0]
    #
    # def normalize(self):
    #     mean, stddev = tf.nn.moments(self._tensor, axes=[0, 1, 2])
    #     self._tensor = (self._tensor - mean) / tf.sqrt(stddev)
    #
    # def add_gaussian_noise(self, stddev=1.):
    #     self._tensor = self._tensor + tf.random.normal(self._tensor.shape, stddev=stddev)

    # def clip(self, min=0., max=np.inf):
    #     self._tensor = tf.clip_by_value(self._tensor, clip_value_min=min, clip_value_max=max)

    #def save(self, path):
    #    np.savez(file=path, tensor=self._tensor.numpy(), extent=self.extent, space=self.space)
