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


def get_neural_net(n_filters=16, dropout=0.1, batchnorm=True):
    input = keras.Input((None, None, 1))

    x = conv2d_block(input, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    # x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    # x = conv2d_block(x, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # x = keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # x = keras.layers.Dropout(dropout)(x)
    # x = conv2d_block(x, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # x = keras.layers.Conv2D(filters=n_filters * 16, kernel_size=(1, 1))(x)
    # x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Conv2D(filters=n_filters * 8, kernel_size=(1, 1))(x)

    # predictors = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), name='markers')(x)

    displacements = keras.layers.Conv2D(filters=2, kernel_size=(1, 1), name='displacements')(x)
    markers = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), name='markers')(x)
    embedding = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), name='embedding')(x)

    # confidence = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), name='confidence')(x)

    return keras.Model(inputs=[input], outputs=[displacements, embedding, markers])


def get_unet(n_filters=16, dropout=0.1, batchnorm=True):
    input = keras.Input((None, None, 1))

    x = conv2d_block(input, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    # x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(dropout)(x)

    # x = conv2d_block(x, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # x = keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # x = keras.layers.Dropout(dropout)(x)
    # x = conv2d_block(x, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    # x = keras.layers.Conv2D(filters=n_filters * 16, kernel_size=(1, 1))(x)
    # x = keras.layers.Dropout(dropout)(x)

    x = keras.layers.Conv2D(filters=n_filters * 8, kernel_size=(1, 1))(x)

    # predictors = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), name='markers')(x)

    displacements = keras.layers.Conv2D(filters=2, kernel_size=(1, 1), name='displacements')(x)
    markers = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), name='markers')(x)
    embedding = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), name='embedding')(x)

    # confidence = keras.layers.Conv2D(filters=1, kernel_size=(1, 1), name='confidence')(x)

    return keras.Model(inputs=[input], outputs=[displacements, embedding, markers])
