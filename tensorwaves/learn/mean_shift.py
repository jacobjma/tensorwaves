import numpy as np
import tensorflow as tf


def adjacent_pairs(n, margin):
    r = tf.range(0, n ** 2, dtype=tf.int64)
    indices = tf.reshape(r, (1, n, n, 1))
    indices = tf.pad(indices, tf.constant([[0, 0],
                                           [margin, margin], [margin, margin], [0, 0]]),
                     mode='CONSTANT', constant_values=-1)
    windows = tf.image.extract_image_patches(indices,
                                             [1, 2 * margin + 1, 2 * margin + 1, 1],
                                             [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')

    pairs = tf.stack((tf.reshape(tf.tile(r[:, None], (1, (2 * margin + 1) ** 2)), (-1,)),
                      tf.reshape(windows, (-1,)),
                      ), -1)

    pairs = tf.boolean_mask(pairs, tf.equal(tf.reduce_any(tf.equal(pairs, -1), axis=1), False))

    return pairs


def sub2ind(rows, cols, n):
    return rows * n + cols


def split_adjacent_pairs(pairs, n):
    lower = tf.boolean_mask(pairs, (pairs[:, 0] - pairs[:, 1]) < 0)

    upper = tf.reverse(lower, (1,))

    linear = sub2ind(upper[:, 0], upper[:, 1], n ** 2)

    lower_to_upper_order = tf.argsort(linear)

    upper = tf.gather(upper, lower_to_upper_order)

    return lower, upper, lower_to_upper_order


def mean_shift_matrix_sparse(pairs, positions, sigma=1):
    n = len(positions)

    values = tf.gather(positions, pairs)
    values = tf.reduce_sum((values[:, 0] - values[:, 1]) ** 2, axis=1)
    values = tf.exp(- .5 * values / sigma ** 2)

    W = tf.sparse.SparseTensor(indices=pairs, values=values, dense_shape=(n, n))
    P = W / tf.sparse.reduce_sum(W, axis=1)[:, None]
    return P


def mean_shift_step_sparse_(pairs, positions, original_positions, eta, sigma=1.):
    n = positions.shape[0]

    values = tf.gather(positions, pairs[:, 0])

    original_positions = tf.gather(original_positions, pairs[:, 1])

    values = tf.reduce_sum((values - original_positions) ** 2, axis=1)

    # print(values.shape, tf.gather(eta, pairs[:, 0])[:, 0].shape)
    # ss
    values = tf.exp(- .5 * values / sigma ** 2)

    # print(values)
    # sss
    values = values * tf.gather(eta, pairs[:, 0])[:, 0] ** 2

    W = tf.sparse.SparseTensor(indices=pairs, values=values, dense_shape=(n, n))

    W = W / tf.sparse.reduce_sum(W, axis=0)

    return tf.sparse.sparse_dense_matmul(W, positions, adjoint_a=True)


def mean_shift_step_sparse(lower, upper, lower_to_upper_order, positions, eta, sigma=1.):
    n = positions.shape[0]

    lower_values = tf.gather(positions, lower)

    lower_values = tf.reduce_sum((lower_values[:, 0] - lower_values[:, 1]) ** 2, axis=1)
    lower_values = tf.exp(- .5 * lower_values / sigma ** 2)

    #lower_values = lower_values * tf.gather(eta, lower[:, 1])[:, 0]

    upper_values = tf.gather(lower_values, lower_to_upper_order)
    #upper_values = upper_values * tf.gather(eta, upper[:, 1])[:, 0]

    W_lower = tf.sparse.SparseTensor(indices=lower, values=lower_values, dense_shape=(n, n))
    W_upper = tf.sparse.SparseTensor(indices=upper, values=upper_values, dense_shape=(n, n))

    col_sum = tf.sparse.reduce_sum(W_lower, axis=1) + tf.sparse.reduce_sum(W_upper, axis=1)
    # return (tf.sparse.sparse_dense_matmul(W_lower, positions) +
    #         tf.sparse.sparse_dense_matmul(W_upper, positions) +
    #         positions) / (col_sum[:, None] + 1)

    return eta * positions + (1 - eta) * (tf.sparse.sparse_dense_matmul(W_lower, positions) +
                                          tf.sparse.sparse_dense_matmul(W_upper, positions) +
                                          positions) / (col_sum[:, None] + 1)

    # return (eta * positions +
    #         (1. - eta) * (tf.sparse.sparse_dense_matmul(W_lower, positions) +
    #                       tf.sparse.sparse_dense_matmul(W_upper, positions) +
    #                       positions) / (col_sum[:, None] + 1))


# def mean_shift_step_sparse(pairs, positions, eta, sigma=1.):
#     P = mean_shift_matrix_sparse(pairs, positions, sigma)
#     return eta * positions + (1. - eta) * tf.sparse.sparse_dense_matmul(P, positions)

def mean_shift_sparse_(positions, eta, num_iter, sigma=1.):
    n = positions.shape[1]

    positions = tf.reshape(positions, (n ** 2, 2))
    eta = tf.reshape(eta, (n ** 2, 1))
    original_positions = tf.identity(positions)

    pairs = adjacent_pairs(n, int(4 * sigma))
    # lower, upper, lower_to_upper_order = split_adjacent_pairs(pairs, n)

    for i in range(num_iter):
        positions = mean_shift_step_sparse_(pairs, positions, original_positions, eta,
                                            sigma=sigma)
    return positions


def mean_shift_sparse(positions, eta, num_iter, sigma=1.):
    n = positions.shape[1]

    positions = tf.reshape(positions, (n ** 2, -1))
    eta = tf.reshape(eta, (n ** 2, 1))

    pairs = adjacent_pairs(n, int(3 * sigma))
    lower, upper, lower_to_upper_order = split_adjacent_pairs(pairs, n)

    for i in range(num_iter):
        positions = mean_shift_step_sparse(lower, upper, lower_to_upper_order, positions, eta, sigma=sigma)
    return positions


def mean_shift_loss_sparse_(positions_, positions, weights, eta, num_iter, sigma=1):
    loss = 0.
    n = positions.shape[1]

    positions_ = tf.reshape(positions_, (-1, n ** 2, 2))
    positions = tf.reshape(positions, (-1, n ** 2, 2))
    original_positions = tf.identity(positions)

    weights = tf.reshape(weights, (-1, n ** 2, 1))
    eta = tf.reshape(eta, (-1, n ** 2, 1))

    pairs = adjacent_pairs(n, int(sigma * 3))

    for i in range(positions_.shape[0]):
        shifted_positions = positions[i]

        for j in range(num_iter):
            loss += tf.reduce_sum((shifted_positions - positions_[i]) ** 2 * weights[i])

            shifted_positions = mean_shift_step_sparse_(pairs, shifted_positions,
                                                        original_positions[i],
                                                        eta[i], sigma=sigma)

    return loss / num_iter


def mean_shift_loss_sparse(positions_, positions, weights, eta, num_iter, sigma=1):
    loss = 0.
    n = positions.shape[1]

    positions_ = tf.reshape(positions_, (-1, n ** 2, 2))
    positions = tf.reshape(positions, (-1, n ** 2, 2))
    weights = tf.reshape(weights, (-1, n ** 2, 1))
    eta = tf.reshape(eta, (-1, n ** 2, 1))

    pairs = adjacent_pairs(n, int(sigma * 3))
    lower, upper, lower_to_upper_order = split_adjacent_pairs(pairs, n)

    for i in range(positions_.shape[0]):
        shifted_positions = positions[i]

        for j in range(num_iter):
            loss += tf.reduce_sum((shifted_positions - positions_[i]) ** 2 * weights[i])

            shifted_positions = mean_shift_step_sparse(lower, upper, lower_to_upper_order, shifted_positions,
                                                       eta[i], sigma=sigma)

    return loss / num_iter


def mean_shift_step(positions, eta, sigma=.5):
    P = mean_shift_matrix(positions, sigma)
    return eta * positions + (1. - eta) * tf.matmul(P, positions, transpose_a=True)


def pairwise_squared_distance(x):
    r = tf.reduce_sum(x * x, 1)
    r = tf.reshape(r, [-1, 1])
    return r - 2 * tf.matmul(x, tf.transpose(x)) + tf.transpose(r)


def mean_shift_matrix(positions, sigma=1.):
    W = -tf.exp(- .5 * pairwise_squared_distance(positions) / sigma ** 2)
    P = tf.matmul(W, tf.linalg.diag(1. / (tf.reduce_sum(W, axis=0))))
    return P


def mean_shift(positions, eta, num_iter, sigma=.5):
    for i in range(num_iter):
        positions = mean_shift_step(positions, eta, sigma=sigma)
    return positions


def mean_shift_loss(label_positions, positions, weights, eta, num_iter):
    loss = 0.
    for i in range(label_positions.shape[0]):
        tmp_positions = positions[i]
        for j in range(num_iter):
            loss += tf.reduce_sum((tmp_positions - label_positions[i]) ** 2 * weights[i])
            tmp_positions = mean_shift_step(tmp_positions, eta[i])
    return loss / num_iter
