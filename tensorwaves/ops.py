import tensorflow as tf


def squared_norm(x, y):
    return x[:, None] ** 2 + y[None, :] ** 2


def angle(x, y):
    return tf.atan2(x[:, None], y[None, :])
