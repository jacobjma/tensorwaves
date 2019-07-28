import tensorflow as tf
from tensorwaves.learn.preprocessing import pad_to_closest_multiple, normalize_global, rescale
from tensorwaves.utils import BatchGenerator
from tensorwaves.utils import create_progress_bar
import numpy as np


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal",
                               padding="same")(input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                               kernel_initializer="he_normal",
                               padding="same")(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def get_unet(x, down_levels=4, up_levels=4, n_filters=16, dropout=0.2, batchnorm=True):
    down = []
    up = []

    for i in range(down_levels):
        x = conv2d_block(x, n_filters=n_filters * 2 ** i, kernel_size=3, batchnorm=batchnorm)
        down.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = conv2d_block(x, n_filters=n_filters * 2 ** down_levels, kernel_size=3, batchnorm=batchnorm)
    bridge = x

    for i in range(up_levels):
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        x = tf.keras.layers.concatenate([x, down[-(i + 1)]])
        x = tf.keras.layers.Dropout(dropout)(x)
        x = conv2d_block(x, n_filters=n_filters * 2 ** (down_levels - i - 1), kernel_size=3, batchnorm=batchnorm)
        up.append(x)

    return down, bridge, up


def get_model(n_classes=None, down_levels=4, up_levels=4, n_filters=16, dropout=0.2, batchnorm=True, use_confidence=True):
    input_img = tf.keras.layers.Input((None, None, 1), name='img')

    down, bridge, up = get_unet(input_img, down_levels=down_levels, up_levels=up_levels, n_filters=n_filters,
                                dropout=dropout,
                                batchnorm=batchnorm)

    markers = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(up[-1])

    if use_confidence:
        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(up[-2])
        markers_confidence = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    if n_classes is not None:
        class_indicator = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='softmax')(up[-1])

        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')(up[-2])
        class_confidence = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)


        return tf.keras.Model(inputs=[input_img], outputs=[markers, class_indicator,
                                                           markers_confidence, class_confidence])
    else:
        if use_confidence:
            return tf.keras.Model(inputs=[input_img], outputs=[markers, markers_confidence])
        else:
            return tf.keras.Model(inputs=[input_img], outputs=[markers])


def train_step(input_image, targets, class_weights, model, optimizer):
    markers_, class_indicators_ = targets

    with tf.GradientTape() as tape:
        markers, class_indicators, markers_confidence, class_confidence = model(input_image, training=True)

        markers = markers_confidence * markers + (1 - markers_confidence) * markers_

        # markers_loss = tf.reduce_mean((markers_ - markers) ** 2)
        markers_loss = tf.reduce_mean(tf.abs(markers_ - markers))

        class_indicators = class_confidence * class_indicators + (1 - class_confidence) * class_indicators_

        class_indicator_loss = tf.reduce_mean(
            categorical_crossentropy(class_indicators_, class_indicators) * class_weights[..., 0])

        # markers_confidence_loss = tf.reduce_mean((1 - markers_confidence) ** 2)
        markers_confidence_loss = tf.reduce_mean(tf.abs(1 - markers_confidence))

        class_confidence_loss = - tf.reduce_mean(tf.math.log(class_confidence))

        confidence_lmbda = .01

        loss = ((markers_loss + class_indicator_loss) +
                confidence_lmbda * (markers_confidence_loss + class_confidence_loss))

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, markers_loss, class_indicator_loss, markers_confidence_loss, class_confidence_loss


def train_step_no_classes(input_image, targets, model, optimizer):
    markers_ = targets[0]

    with tf.GradientTape() as tape:
        markers, markers_confidence = model(input_image, training=True)

        markers = markers_confidence * markers + (1 - markers_confidence) * markers_

        markers_loss = tf.reduce_mean((markers_ - markers) ** 2)

        markers_confidence_loss = tf.reduce_mean((1 - markers_confidence) ** 2)

        confidence_lmbda = .2

        loss = markers_loss + confidence_lmbda * markers_confidence_loss

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, markers_loss, markers_confidence_loss


class AtomRecognitionModel(object):

    def __init__(self, neural_net, model_sampling, compression, scale_estimator, margin=0, classes=False):
        self._neural_net = neural_net
        self._model_sampling = model_sampling
        self._compression = compression
        self._scale_estimator = scale_estimator
        self._margin = margin
        self._classes = classes

    def preprocess(self, sequence):
        sequence = tf.constant(sequence, dtype=tf.float32)

        if len(sequence.shape) == 2:
            sequence = tf.expand_dims(tf.expand_dims(sequence, -1), 0)

        elif len(sequence.shape) == 3:
            sequence = tf.expand_dims(sequence, -1)

        elif len(sequence.shape) != 4:
            raise RuntimeError('')

        sampling = self._scale_estimator.estimate_scale(sequence[0])

        scale = sampling / self._model_sampling

        sequence = normalize_global(rescale(sequence, scale))

        input_shape = (int(np.ceil((sequence.shape[1] + 2 * self._margin) / 16) * 16),
                       int(np.ceil((sequence.shape[2] + 2 * self._margin) / 16) * 16))

        padding = (input_shape[0] - sequence.shape[1], input_shape[1] - sequence.shape[2])

        shift = (padding[0] // 2, padding[1] // 2)

        sequence = tf.image.pad_to_bounding_box(sequence, shift[0], shift[1], input_shape[0], input_shape[1])

        valid = (shift[0] // self._compression, (input_shape[0] - padding[0] // 2) // self._compression,
                 shift[1] // self._compression, (input_shape[1] - padding[1] // 2) // self._compression)

        out_shape = (sequence.shape[1] // self._compression, sequence.shape[2] // self._compression)

        shift = (shift[0] / scale * sampling, shift[1] / scale * sampling)

        return sequence, out_shape, valid, self._compression / scale * sampling, shift, sampling

    def predict(self, sequence, batch_size=1, progress_bar=True, dropout=False):

        if not self._classes:
            return self._predict_no_classes(sequence, batch_size=batch_size, progress_bar=progress_bar, dropout=dropout)

        else:
            pass

    # def _predict_classes(self, sequence, batch_size=1, sampling=None, progress_bar=True, dropout=False):
    #     sequence, out_shape, valid, scale, shift, sampling = self.preprocess(sequence, sampling=sampling)
    #
    #     class_indicators = np.zeros((sequence.shape[0],) + out_shape + (2,))
    #     class_confidence = np.zeros((sequence.shape[0],) + out_shape + (1,))
    #     class_indicators[i: i + n] = batch_class_indicators
    #     class_confidence[i: i + n] = batch_class_confidence
    #     batch_markers, batch_class_indicators, batch_markers_confidence, batch_class_confidence = self._neural_net(
    #         sequence[i: i + n], training=dropout)

    def _predict_no_classes(self, sequence, batch_size=1, progress_bar=True, dropout=False):
        sequence, out_shape, valid, scale, shift, sampling = self.preprocess(sequence)

        markers = np.zeros((sequence.shape[0],) + out_shape + (1,))
        confidence = np.zeros((sequence.shape[0],) + out_shape + (1,))

        generator = BatchGenerator(len(sequence), batch_size)

        for i, n in create_progress_bar(generator.generate(), num_iter=generator.n_batches, description='Batch #:',
                                        disable=not progress_bar):
            batch_markers, batch_confidence = self._neural_net(sequence[i: i + n], training=dropout)
            markers[i: i + n] = batch_markers
            confidence[i: i + n] = batch_confidence

        return markers, confidence, valid, scale, shift, sampling
