import numpy as np
import tensorflow as tf
from tensorflow import keras


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                            padding="same")(input_tensor)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation("relu")(x)

    x = keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                            padding="same")(x)
    if batchnorm:
        x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Activation("relu")(x)
    return x


def get_unet(out_channels, n_filters=16, dropout=0.5, batchnorm=True, train=True):
    images = keras.Input((None, None, 1))
    if train:
        labels = keras.Input((None, None, out_channels))
    else:
        labels = None

    # contracting path
    c1 = conv2d_block(images, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = keras.layers.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = keras.layers.Dropout(dropout)(p3)
    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c4)
    u7 = keras.layers.concatenate([u7, c3])
    u7 = keras.layers.Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = keras.layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    u8 = keras.layers.Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    u9 = keras.layers.Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    if out_channels == 1:
        predictions = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='predictions')(c9)
    else:
        predictions = keras.layers.Conv2D(out_channels, (1, 1), activation='softmax', name='predictions')(c9)

    confidence = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='confidence')(c9)

    if train:
        def corrected_predictions_func(tensors):
            predictions, labels, confidence = tensors
            return confidence * predictions + (1 - confidence) * labels

        corrected_predictions = keras.layers.Lambda(corrected_predictions_func, name='corrected_predictions')(
            [predictions, labels, confidence])

        def dummy(tensors):
            return tensors[0]

        counts = keras.layers.Lambda(dummy, name='counts')([corrected_predictions])

        return keras.Model(inputs=[images, labels], outputs=[corrected_predictions, confidence, counts])
    else:
        return keras.Model(inputs=[images], outputs=[predictions, confidence])


_epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), tf.float32)


def pixel_weights_func(labels):
    weights = tf.reduce_sum(labels, axis=(1, 2), keepdims=True)
    weights /= tf.reduce_sum(weights, axis=3, keepdims=True)
    weights = (weights + 1e-3)
    return tf.reduce_sum(labels / weights, axis=3, keepdims=True)


def xentropy_loss(labels, predictions):
    predictions /= tf.reduce_sum(predictions, axis=-1, keepdims=True)
    predictions = tf.clip_by_value(predictions, clip_value_min=_epsilon, clip_value_max=1. - _epsilon)
    # pixel_weights = pixel_weights_func(labels)
    loss = - tf.reduce_sum(labels * tf.math.log(predictions), axis=-1, keepdims=True)  # * pixel_weights
    return tf.reduce_mean(loss, axis=(1, 2, 3))


def xentropy_budget_loss(labels, predictions):
    predictions = tf.clip_by_value(predictions, clip_value_min=_epsilon, clip_value_max=1. - _epsilon)
    # pixel_weights = pixel_weights_func(labels)
    loss = (- tf.math.log(predictions))  # * pixel_weights
    return tf.reduce_mean(loss, axis=(1, 2, 3))  # + .5e-4 * tf.image.total_variation(predictions)


def mse_loss(labels, predictions):
    return tf.reduce_mean((labels - predictions) ** 2, axis=-1, keepdims=True)


def mse_patch_loss(labels, predictions):
    labels = tf.nn.avg_pool(labels, [1, 32, 32, 1], [1, 32, 32, 1], 'VALID')
    predictions = tf.nn.avg_pool(predictions, [1, 32, 32, 1], [1, 32, 32, 1], 'VALID')
    return tf.reduce_mean(labels - predictions, axis=-1, keepdims=True) ** 2


def gaussian_kernel(sigma, dtype=tf.float32):
    truncate = tf.math.ceil(3 * tf.cast(sigma, dtype))
    x = tf.cast(tf.range(-truncate, truncate + 1), dtype)
    return 1. / (np.sqrt(2. * np.pi) * sigma) * tf.exp(-x ** 2 / (2 * sigma ** 2))


def gaussian_filter(image, sigma):
    kernel = gaussian_kernel(sigma)
    image = tf.nn.conv2d(image, kernel[..., None, None, None], strides=[1, 1, 1, 1], padding='SAME')
    image = tf.nn.conv2d(image, kernel[None, ..., None, None], strides=[1, 1, 1, 1], padding='SAME')
    return image


def mse_budget_loss(labels, predictions):
    base_loss = tf.reduce_mean((1 - predictions), axis=-1, keepdims=True)
    smoothed = gaussian_filter(predictions, 10)
    reg_loss = tf.abs(predictions - smoothed)
    return base_loss + 2 * reg_loss


class BudgetTuner(tf.keras.callbacks.Callback):
    def __init__(self, lmbda, name, budget=.3):
        self.lmbda = lmbda
        self.budget = budget
        self.name = name

    def on_batch_end(self, batch, logs={}):
        if self.budget > logs[self.name + '_loss']:
            self.lmbda.assign(self.lmbda / 1.001)
        else:
            self.lmbda.assign(self.lmbda / 0.999)
