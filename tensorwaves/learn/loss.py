import tensorflow as tf


def binary_classification_pixel_weights(labels):
    positive = labels / tf.reduce_sum(labels, axis=(1, 2, 3), keepdims=True)
    negative = (1 - labels) / tf.reduce_sum(1 - labels, axis=(1, 2, 3), keepdims=True)
    weights = positive + negative
    return weights


def xentropy_loss(labels, predictions):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)
    return tf.reduce_sum(loss)


def weighted_absolute_error(labels, predictions, weights):
    return tf.reduce_sum(tf.abs(labels - predictions) * weights)


def displacements_to_positions(displacements):
    X, Y = tf.meshgrid(tf.range(0, displacements.shape[1]),
                       tf.range(0, displacements.shape[2]), indexing='ij')
    indices = tf.cast(tf.stack((X, Y), -1)[None], tf.float32)
    # displacements = tf.reshape(displacements, (displacements.shape[0], -1, 2))
    positions = indices + displacements
    return positions


def polar_to_displacements(polar):
    x = (2 * tf.nn.sigmoid(polar[..., 2]) - 1)
    y = (2 * tf.nn.sigmoid(polar[..., 1]) - 1)
    r = tf.sqrt(x ** 2 + y ** 2)
    return tf.stack([polar[..., 0] * x / r, polar[..., 0] * y / r], axis=-1)


def generate_indices(labels):
    labels = tf.reshape(labels, (-1,))
    order = tf.argsort(labels)
    sorted_labels = tf.gather(labels, order)
    indices = tf.gather(tf.range(0, labels.shape[0] + 1), order)
    index = tf.range(0, tf.reduce_max(labels) + 1)
    lo = tf.searchsorted(sorted_labels, index, side='left')
    hi = tf.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield tf.sort(indices[l:h])


def embedding_loss(labels, embedding, dd=1.5, dv=.5, gamma=.0001):
    C = tf.reduce_max(labels)
    m = embedding.shape[-1]

    centers = tf.zeros((C, m))
    variances = tf.zeros((C, m))

    labels = tf.reshape(labels, (-1,))
    embedding = tf.reshape(embedding, (-1, m))

    # print(embedding.shape, labels.shape)

    for i, indices in enumerate(generate_indices(labels)):
        values = tf.gather(embedding, indices)
        center = tf.reduce_mean(values, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.maximum(tf.sqrt((values - center + 1.e-5) ** 2 + 1e-5) - dv, 0.), axis=0,
                                  keepdims=True)

        centers = tf.tensor_scatter_nd_update(centers, [[i]], center)
        variances = tf.tensor_scatter_nd_update(variances, [[i]], variance)

    def pairwise_squared_distance(x):
        r = tf.reduce_sum(x * x, 1)
        r = tf.reshape(r, [-1, 1])
        return tf.maximum(r + (- 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)), 1.e-5)

    distances = tf.sqrt(pairwise_squared_distance(centers))

    distance_loss = (tf.reduce_sum(tf.abs(tf.maximum(2 * dd - distances, 1.e-5)))) / (
            tf.cast(C, tf.float32) * (tf.cast(C, tf.float32) - 1))
    # (2. * dd) ** 2 * tf.cast(C, tf.float32)) /

    variance_loss = tf.reduce_mean(variances)

    reg_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.abs(centers))))

    # print(distance_loss, variance_loss, reg_loss)

    loss = distance_loss + variance_loss + gamma * reg_loss

    return loss, distance_loss, variance_loss, reg_loss


def adjacent_embedding_loss(labels, embedding, adjacent, weights, dd=1.5, dv=.5, gamma=.0001):
    C = tf.reduce_max(labels) + 1
    m = embedding.shape[-1]

    centers = tf.zeros((C, m))
    variances = tf.zeros((C, m))

    labels = tf.reshape(labels, (-1,))
    embedding = tf.reshape(embedding, (-1, m))

    for i, indices in enumerate(generate_indices(labels)):

        values = tf.gather(embedding, indices)
        center = tf.reduce_mean(values, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.maximum(tf.sqrt((values - center + 1.e-5) ** 2) - dv, 0.) ** 2, axis=0,
                                  keepdims=True)

        centers = tf.tensor_scatter_nd_update(centers, [[i]], center)
        variances = tf.tensor_scatter_nd_update(variances, [[i]], variance)

    distances = tf.reduce_sum((centers[1:, None, :] - tf.gather(centers, adjacent)) ** 2, axis=-1)

    distance_loss = tf.reduce_mean(tf.maximum(2 * dd - distances, 1.e-5) ** 2 * weights)

    variance_loss = tf.reduce_mean(variances)

    reg_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(centers ** 2)))

    loss = distance_loss + variance_loss + gamma * reg_loss

    return loss, distance_loss, variance_loss, reg_loss
