import numpy as np
import tensorflow as tf




class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, image, label=None, tag_prefix='', log_dir='./logs/images'):
        tf.keras.callbacks.Callback.__init__(self)

        self.tag_prefix = tag_prefix
        self.log_dir = log_dir
        self.image = image
        self.label = label

    def on_epoch_end(self, epoch, logs=None):
        prediction, confidence = self.model.predict((self.image, self.label))
        prediction = np.concatenate([prediction[..., i] for i in range(prediction.shape[-1])], axis=1)[..., None]

        if self.label is not None:
            label = np.concatenate([self.label[..., i] for i in range(self.label.shape[-1])], axis=1)[..., None]
            prediction = np.concatenate((prediction, label), axis=2)

        writer = tf.summary.create_file_writer(self.log_dir)

        with writer.as_default(), tf.summary.record_if(True):
            image = (self.image - self.image.min()) / (self.image.max() - self.image.min())

            tf.summary.image('image', (255 * image).astype(np.uint8), step=epoch)
            tf.summary.image('confidence', (255 * confidence).astype(np.uint8), step=epoch)
            tf.summary.image('prediction', (255 * prediction).astype(np.uint8), step=epoch)

        writer.close()
