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


def get_unet(n_filters=16, dropout=0.5, batchnorm=True):
    input_img = keras.Input((None, None, 1))

    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = keras.layers.Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)
    p2 = keras.layers.Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)
    p3 = keras.layers.Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = keras.layers.Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = keras.layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    u6 = keras.layers.Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = keras.layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
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

    outputs = keras.layers.Conv2D(3, (1, 1), activation='softmax')(c9)
    model = keras.Model(inputs=[input_img], outputs=[outputs])
    return model


def get_weighted_pixelwise_crossentropy(class_weights=1):
    # if isinstance(class_weights, numbers.Number):
    #    pass
    # else:
    #    class_weights = np.append(class_weights, 1)[None, None, None, :]

    weigths = tf.convert_to_tensor([1, 2], dtype=tf.float32)[None, None, None, :]

    def weighted_pixelwise_crossentropy(target, output):
        output /= tf.reduce_sum(output, -1, True)

        _epsilon = tf.convert_to_tensor(keras.backend.epsilon(), output.dtype.base_dtype)

        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

        # shape = output.get_shape().as_list()
        # shape[-1] = 1

        pixel_weights = 1 + tf.reduce_sum(target[:, :, :, :-1] * weigths, axis=-1, keepdims=True)

        # pixel_weights = 1 + target[..., 0, None] + 2 * target[..., 1, None]

        return - tf.reduce_sum(tf.multiply(target * tf.log(output), pixel_weights), -1)

    return weighted_pixelwise_crossentropy
