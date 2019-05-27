import numpy as np
import tensorflow as tf

from tensorwaves.transfer import PrismTranslate


class DataGenerator(object):
    def __init__(self, positions, values, batch_size=32):

        self._batch_size = batch_size

        def generator(positions, values, batch_size):
            num_iter = len(values) // batch_size
            while True:
                for i in range(num_iter):
                    if i == 0:
                        indices = np.arange(len(values))
                        np.random.shuffle(indices)

                    batch_indices = indices[i * batch_size:(i + 1) * batch_size]

                    batch_values = values[batch_indices]
                    batch_positions = positions[batch_indices]

                    yield batch_indices, batch_positions, batch_values

        self._gen = generator(positions, values, batch_size=batch_size)
        self._length = len(values)
        self._values = values
        self._positions = positions

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def values(self):
        return self._values

    @property
    def positions(self):
        return self._values

    def __len__(self):
        return self._length

    def __iter__(self):
        return self._gen

    def __next__(self):
        return next(self._gen)


def gaussian(r2, a, b):
    return a * tf.exp(-r2 / (b ** 2))


def gaussian_derivative(r, r2, a, b):
    return - 2 * a * 1 / b ** 2 * r * tf.exp(-r2 / b ** 2)


def soft_gaussian(r, r2, a, b, r_cut):
    return (gaussian(r2, a, b) - gaussian(r_cut ** 2, a, b) - (r - r_cut) *
            gaussian_derivative(r_cut, r_cut ** 2, a, b))


def wrapped_slice(tensor, begin, size):
    shift = [-x for x in begin]
    tensor = tf.roll(tensor, shift, list(range(len(begin))))
    return tf.slice(tensor, [0] * len(begin), size)


class Killswitch(object):

    def __init__(self):
        self.on = False


class ProbeModel(object):

    def __init__(self, S, detector, scale, radius):
        self._S = S

        detector.extent = S.probe_extent
        detector.gpts = S.probe_gpts
        detector.energy = S.energy

        self._detector = detector.build().tensor()[0]
        self._translate = PrismTranslate(kx=S.kx, ky=S.ky)

        alpha_x = self.S.kx * self.S.wavelength
        alpha_y = self.S.ky * self.S.wavelength
        self._alpha2 = alpha_x ** 2 + alpha_y ** 2

        self._scale = tf.Variable(scale, dtype=tf.float32)
        self._radius = tf.Variable(radius, dtype=tf.float32)
        self._losses = []
        self._last_predictions = None

    @property
    def S(self):
        return self._S

    @property
    def translate(self):
        return self._translate

    def get_coefficients(self):
        scale = self._scale
        radius = tf.abs(self._radius)
        return tf.reduce_sum(tf.cast(gaussian(self._alpha2[:, None], scale[None, :], radius[None, :]), tf.complex64),
                             axis=1)

    def get_probe(self, position, coefficients):

        # begin = [
        #     np.round((position[0] - self.S.extent[0] / (2 * self.S.interpolation)) /
        #              self.S.sampling[0]).astype(int),
        #     np.round((position[1] - self.S.extent[1] / (2 * self.S.interpolation)) /
        #              self.S.sampling[1]).astype(int)
        # ]
        #
        # size = [0,
        #     np.ceil(self.S.gpts[0] / self.S.interpolation).astype(int),
        #     np.ceil(self.S.gpts[1] / self.S.interpolation).astype(int)
        # ]

        # tensor = self.S._expansion[
        #          :,
        #          begin[0]:begin[0] + size[0],
        #          begin[1]:begin[1] + size[1]
        #          ]
        begin = [0,
                 np.round((position[0] - self.S.extent[0] / (2 * self.S.interpolation)) /
                          self.S.sampling[0]).astype(int),
                 np.round((position[1] - self.S.extent[1] / (2 * self.S.interpolation)) /
                          self.S.sampling[1]).astype(int)]

        size = [self.S.kx.shape[0],
                np.ceil(self.S.gpts[0] / self.S.interpolation).astype(int),
                np.ceil(self.S.gpts[1] / self.S.interpolation).astype(int)]

        tensor = wrapped_slice(self.S._expansion, begin, size)

        # tensor = self.S._expansion[
        #          :,
        #          begin[0]:begin[0] + size[0],
        #          begin[1]:begin[1] + size[1]
        #          ]

        self.translate.positions = position

        probe = tf.reduce_sum(tensor * (self.translate.tensor() * coefficients)[:, None, None], axis=0)

        return probe

    def predict(self, position, coefficients):
        probe = self.get_probe(position, coefficients)

        diffraction = tf.abs(tf.signal.fft2d(probe)) ** 2
        y_predict = tf.reduce_sum(tf.boolean_mask(diffraction, self._detector))

        return y_predict / tf.reduce_sum(diffraction)

    def fit_generator(self, data_generator, num_epochs, optimizers, max_position=None, callback=None, killswitch=None):

        steps_per_epoch = len(data_generator)

        self._last_predictions = np.zeros(len(data_generator))

        for epoch in range(num_epochs):
            epoch_loss = 0.

            for i in range(steps_per_epoch):
                indices, X_batch, Y_batch = next(data_generator)

                with tf.GradientTape() as tape:
                    batch_loss = 0.
                    coefficients = self.get_coefficients()

                    max_value = self.predict(max_position, coefficients)
                    # print(max_value)

                    for j, (x, y) in enumerate(zip(X_batch, Y_batch)):
                        y_predict = self.predict(x, coefficients) / max_value
                        batch_loss += tf.reduce_sum(tf.square(y - y_predict))
                        self._last_predictions[indices[j]] = y_predict

                    batch_loss /= len(X_batch)

                epoch_loss += batch_loss
                # running_mean = epoch_loss / ((i + 1) * data_generator.batch_size)

                self._losses.append(batch_loss)

                grads = tape.gradient(batch_loss, [self._scale, self._radius])

                clip = .01 / optimizers[0].learning_rate
                grads[0] = tf.clip_by_value(grads[0], -clip, clip)

                clip = .0001 / optimizers[1].learning_rate
                grads[1] = tf.clip_by_value(grads[1], -clip, clip)

                optimizers[0].apply_gradients(zip([grads[0]], [self._scale]))
                optimizers[1].apply_gradients(zip([grads[1]], [self._radius]))

                #fwhm = get_fwhm(p[p.shape[0] // 2], self.S.probe_extent[1])

                #print('Step: {}, Batch loss: {:.3e}, FWHM: {:.3f}'.format(i, batch_loss, fwhm))

                if callback is not None:
                    callback.on_batch_end()

                if killswitch is not None:
                    if killswitch.on:
                        return

                # clear_output(wait=True)
