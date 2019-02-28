import numpy as np
import tensorflow as tf

from tensorwaves.prism import PrismTranslate
from tensorwaves.transfer import zernike2polar
from tensorwaves.utils import complex_exponential


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=32, shuffle=True):

        self.batch_size = batch_size
        assert len(X) == len(Y)
        self.X = X.copy()
        self.Y = Y.copy()
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, batch_index):
        # Generate indexes of the batch
        batch_indices = self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

        if batch_index == len(self) - 1:
            self.on_epoch_end()

        return self.__data_generation(batch_indices)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        np.random.seed(1)
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        return self.X[batch_indices], self.Y[batch_indices], batch_indices


class Adam(object):

    def __init__(self, learning_rate, beta1=0.85, beta2=0.999, epsilon=1e-6):
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._learning_rate = learning_rate

        self._i = 0
        self._t = 0

    def collect_grad(self, grad):
        if self._i == 0:
            self._grad = tf.zeros_like(grad)

        self._grad += grad
        self._i += 1

    def update(self, f):
        if self._t == 0:
            self._v = tf.zeros_like(f, dtype=tf.float32)
            self._sqr = tf.zeros_like(f, dtype=tf.float32)

        grad = self._grad / self._i
        self._i = 0
        self._t += 1

        v = self._beta1 * self._v + (1. - self._beta1) * grad
        sqr = self._beta2 * self._sqr + (1. - self._beta2) * tf.square(grad)

        v_bias_corr = v / (1. - self._beta1 ** self._t)
        sqr_bias_corr = sqr / (1. - self._beta2 ** self._t)
        update = self._learning_rate * v_bias_corr / (tf.sqrt(sqr_bias_corr) + self._epsilon)

        f.assign_sub(update)


class ProbeModel(object):

    def __init__(self, S, detector):
        self._S = S
        self._detector = detector

        self._translate = PrismTranslate(kx=S.kx, ky=S.ky)

        aberrations = S.aberrations.parametrization.to_zernike(.035)

        alpha_x, alpha_y = S.alpha_x.numpy(), S.alpha_y.numpy()
        alpha = np.sqrt(alpha_x ** 2 + alpha_y ** 2)
        phi = np.arctan2(alpha_x, alpha_y)

        self._indices, self._basis, self._expansion = aberrations.expansion(alpha, phi)

        self._expansion = tf.Variable(self._expansion)
        self._scale = tf.Variable(tf.convert_to_tensor([1.], dtype=tf.float32))
        self._shift = tf.Variable(tf.convert_to_tensor([0.], dtype=tf.float32))

    def get_coefficients(self):
        chi = tf.reduce_sum(self._basis * self._expansion[:, None], axis=0)
        coefficients = complex_exponential(- 2 * np.pi / self._S.wavelength * chi)
        return coefficients

    def get_probe(self):
        self._translate.position = (0., 0.)
        coefficients = self._translate.get_tensor() * self.get_coefficients()
        return tf.abs(tf.reduce_sum(
            self._S._tensorflow * coefficients[:, None, None] *
            tf.cast(self._S.aperture.get_tensor()[:, None, None], dtype=tf.complex64), axis=0)) ** 2

    def evaluate_loss(self, x, y):
        y_predict = self.predict(x)  # * self._linear_transform[0] + self._linear_transform[1]
        return tf.square(y_predict - y), y_predict

    def predict(self, x):
        self._translate.position = x
        coefficients = self._translate.get_tensor() * self.get_coefficients()

        # print(self._S.aperture.get_tensor())

        probe = tf.reduce_sum(
            self._S._tensorflow * coefficients[:, None, None] *
            tf.cast(self._S.aperture.get_tensor()[:, None, None], dtype=tf.complex64), axis=0)

        diffraction = tf.abs(tf.fft2d(probe)) ** 2
        y_predict = tf.reduce_sum(diffraction * self._detector.get_tensor()) / tf.reduce_sum(diffraction)
        return y_predict

    def fit_generator(self, data_generator, num_epochs, learning_rate, callback=None):

        expansion_optimizer = Adam(learning_rate)
        # scale_optimizer = Adam(.1)

        for i, (X_batch, Y_batch, batch_indices) in enumerate(data_generator):
            batch_predict = tf.zeros([data_generator.batch_size, 1])
            batch_y = tf.zeros([data_generator.batch_size, 1])

            with tf.GradientTape() as tape:
                for j, (x, y, index) in enumerate(zip(X_batch, Y_batch, batch_indices)):
                    y_predict = self.predict(x)  # * self._scale  # - mean_old
                    batch_predict += tf.scatter_nd([[j]], y_predict[None, None], batch_predict.shape)
                    batch_y += tf.scatter_nd([[j]], tf.convert_to_tensor(y)[None, None], batch_predict.shape)

                mean, var = tf.nn.moments(batch_predict, axes=[0, 1])
                batch_predict = (batch_predict - mean) / tf.sqrt(var)

                #mean, var = tf.nn.moments(batch_y, axes=[0, 1])
                #batch_y = (batch_y - mean) / tf.sqrt(var)

                batch_loss = tf.reduce_sum(tf.square(batch_predict - batch_y))

            expansion_grad = tape.gradient(batch_loss, [self._expansion, ])  # self._scale])

            expansion_optimizer.collect_grad(expansion_grad)
            # scale_optimizer.collect_grad(scale_grad)
            # shift_optimizer.collect_grad(shift_grad)

            expansion_optimizer.update(self._expansion)
            # scale_optimizer.update(self._scale)
            # shift_optimizer.update(self._shift)

            if callback is not None:
                expansion = dict(zip(self._indices, list(self._expansion.numpy())))
                expansion = zernike2polar(expansion, .035)
                callback.on_batch_end(i, y_predict, batch_loss, batch_indices, self.get_probe().numpy(), expansion,
                                      self._scale.numpy(), self._shift.numpy())
