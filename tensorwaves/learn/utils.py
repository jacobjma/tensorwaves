from PIL import Image
import io
import tensorflow as tf
import numpy as np


def make_image(tensor):
    tensor = np.tile((255 * (tensor - tensor.min()) /
                      (tensor.max() - tensor.min())).astype('uint8'), (1, 1, 3))

    height, width, channel = tensor.shape
    image = Image.fromarray(tensor)

    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()

    return tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, tag, x, model, log_dir='./logs/images'):
        tf.keras.callbacks.Callback.__init__(self)

        self.tag = tag
        self.log_dir = log_dir
        self.x = x
        #self.y = y
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.x, verbose=1)
        preds = np.concatenate([preds[..., 0], preds[..., 1], preds[..., 2]], axis=2)[0, ..., None]
        #y = np.concatenate([self.y[..., 0], self.y[..., 1], self.y[..., 2]], axis=2)[0, ..., None]

        #y_image = make_image(y)
        x_image = make_image(self.x[0])
        preds_image = make_image(preds)

        summary = tf.Summary(value=[#tf.Summary.Value(tag=self.tag + ' ground truth', image=y_image),
                                    tf.Summary.Value(tag=self.tag + ' image', image=x_image),
                                    tf.Summary.Value(tag=self.tag + ' predictions', image=preds_image)])

        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(summary, epoch)
        writer.close()

        return
