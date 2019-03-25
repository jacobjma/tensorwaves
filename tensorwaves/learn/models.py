from tensorflow import keras
import tensorflow as tf
import numpy as np
import numbers


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


def get_unet(out_channels, n_filters=16, levels=3, dropout=0.5, batchnorm=True, train=True):
    images = keras.Input((None, None, 1))
    if train:
        labels = keras.Input((None, None, out_channels))

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

    def corrected_predictions_func(tensors):
        predictions, labels, confidence = tensors
        return confidence * predictions + (1 - confidence) * labels

    if train:
        corrected_predictions = keras.layers.Lambda(corrected_predictions_func, name='corrected_predictions')(
            [predictions, labels, confidence])

        return keras.Model(inputs=[images, labels], outputs=[corrected_predictions, confidence])
    else:
        return keras.Model(inputs=[images], outputs=[predictions, confidence])


def pixel_weights(labels):
    weights = tf.reduce_sum(labels, axis=(1, 2), keep_dims=True)
    weights /= tf.reduce_sum(weights, axis=3, keep_dims=True)
    weights = weights + 1e-5
    return tf.reduce_sum(labels / weights, axis=3, keep_dims=True)


def get_weighted_pixelwise_crossentropy(pixel_weights_func):
    _epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), tf.float32)

    def weighted_pixelwise_crossentropy(labels, predictions):
        predictions /= tf.reduce_sum(predictions, -1, True)

        predictions = tf.clip_by_value(predictions, _epsilon, 1. - _epsilon)

        pixel_weights = pixel_weights_func(labels)

        return - tf.reduce_sum(tf.multiply(labels * tf.log(predictions), pixel_weights), -1)

    return weighted_pixelwise_crossentropy
