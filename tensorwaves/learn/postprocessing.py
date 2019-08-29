import numba
import numpy as np

from sklearn.neighbors import NearestNeighbors


def stack_uneven(arrays, fill_value=-1, dtype=None):
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value, dtype=dtype)
    for i, a in enumerate(arrays):
        slices = tuple(slice(0, s) for s in sizes[i])
        result[i][slices] = a
    return result


@numba.njit
def sub2ind(rows, cols, array_shape):
    return rows * array_shape[1] + cols


@numba.njit
def ind2sub(array_shape, ind):
    rows = (np.int32(ind) // array_shape[1])
    cols = (np.int32(ind) % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


# @numba.njit
def non_maximum_suppresion(markers, class_indicators, distance, threshold):
    shape = markers.shape[1:-1]

    markers = markers.reshape((markers.shape[0], -1))

    if class_indicators is not None:
        class_indicators = class_indicators.reshape((class_indicators.shape[0], -1, class_indicators.shape[-1]))
        class_probabilities = np.zeros(class_indicators.shape, dtype=class_indicators.dtype)

    accepted = np.zeros(markers.shape, dtype=np.bool_)
    suppressed = np.zeros(markers.shape, dtype=np.bool_)

    x_disc = np.zeros((2 * distance + 1, 2 * distance + 1), dtype=np.int32)

    x_disc[:] = np.linspace(0, 2 * distance, 2 * distance + 1)
    y_disc = x_disc.copy().T
    x_disc -= distance
    y_disc -= distance
    x_disc = x_disc.ravel()
    y_disc = y_disc.ravel()

    r2 = x_disc ** 2 + y_disc ** 2

    x_disc = x_disc[r2 < distance ** 2]
    y_disc = y_disc[r2 < distance ** 2]

    weights = np.exp(-r2 / (2 * (distance / 3) ** 2))
    weights = np.reshape(weights[r2 < distance ** 2], (-1, 1))

    for i in range(markers.shape[0]):
        suppressed[i][markers[i] < threshold] = True
        for j in np.argsort(-markers[i].ravel()):
            if not suppressed[i, j]:
                accepted[i, j] = True

                x, y = ind2sub(shape, j)
                neighbors_x = x + x_disc
                neighbors_y = y + y_disc

                valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (
                        neighbors_y < shape[1]))

                neighbors_x = neighbors_x[valid]
                neighbors_y = neighbors_y[valid]

                k = sub2ind(neighbors_x, neighbors_y, shape)
                suppressed[i][k] = True

                if class_probabilities is not None:
                    tmp = np.sum(class_indicators[i][k] * weights[valid], axis=0)
                    class_probabilities[i, j] = tmp / np.sum(tmp)

    accepted = accepted.reshape((markers.shape[0],) + shape)

    if class_probabilities is not None:
        class_probabilities = class_probabilities.reshape(
            (class_indicators.shape[0],) + shape + (class_indicators.shape[-1],))

        return accepted, class_probabilities

    else:
        return accepted, 1


# @numba.njit
# def non_maximum_suppresion_with_map(markers, threshold, x_disc, y_disc, weights):
#     shape = markers.shape
#
#     markers = markers.ravel()
#     accepted = np.zeros(markers.shape, dtype=np.bool_)
#     suppressed = np.zeros(markers.shape, dtype=np.bool_)
#
#     suppressed[markers < threshold] = True
#     for j in np.argsort(-markers.ravel()):
#         if not suppressed[j]:
#             accepted[j] = True
#
#             x, y = ind2sub(shape, j)
#             neighbors_x = x + x_disc
#             neighbors_y = y + y_disc
#
#             valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (neighbors_y < shape[1]))
#
#             neighbors_x = neighbors_x[valid]
#             neighbors_y = neighbors_y[valid]
#
#             k = sub2ind(neighbors_x, neighbors_y, shape)
#             suppressed[k] = True
#
#     accepted = accepted.reshape(shape)
#
#     return accepted


# @numba.njit
def non_maximum_suppresion(markers, threshold, x_disc, y_disc, weights=None, maps=None):
    shape = markers.shape

    if maps is not None:
        maps = maps.reshape((-1, maps.shape[-1]))
        assigned = np.zeros(maps.shape, dtype=maps.dtype)
    else:
        assigned = None

    markers = markers.ravel()
    accepted = np.zeros(markers.shape, dtype=np.bool_)
    suppressed = np.zeros(markers.shape, dtype=np.bool_)

    suppressed[markers < threshold] = True
    for i in np.argsort(-markers.ravel()):
        if not suppressed[i]:
            accepted[i] = True

            x, y = ind2sub(shape, i)
            neighbors_x = x + x_disc
            neighbors_y = y + y_disc

            valid = ((neighbors_x > -1) & (neighbors_y > -1) & (neighbors_x < shape[0]) & (neighbors_y < shape[1]))

            neighbors_x = neighbors_x[valid]
            neighbors_y = neighbors_y[valid]

            k = sub2ind(neighbors_x, neighbors_y, shape)
            suppressed[k] = True

            if assigned is not None:
                assigned[i] = np.sum(maps[k][:] * weights[valid], axis=0)

    if assigned is not None:
        assigned = assigned.reshape(shape + (assigned.shape[-1],))

    accepted = accepted.reshape(shape)

    return accepted, assigned


@numba.njit
def _index_disc(distance):
    x_disc = np.zeros((2 * distance + 1, 2 * distance + 1), dtype=np.int32)

    x_disc[:] = np.linspace(0, 2 * distance, 2 * distance + 1)
    y_disc = x_disc.copy().T
    x_disc -= distance
    y_disc -= distance
    x_disc = x_disc.ravel()
    y_disc = y_disc.ravel()

    r2 = x_disc ** 2 + y_disc ** 2

    x_disc = x_disc[r2 < distance ** 2]
    y_disc = y_disc[r2 < distance ** 2]

    weights = np.exp(-r2 / (2 * (distance / 4) ** 2))
    weights = np.reshape(weights[r2 < distance ** 2], (-1, 1))

    weights = weights / weights.sum()

    return x_disc, y_disc, weights


class NMS(object):

    def __init__(self, distance, threshold, return_indices=True):
        self._distance = distance
        self._threshold = threshold

        self._return_indices = return_indices

        self._x_disc, self._y_disc, self._weights = _index_disc(distance)

    def run(self, markers, *maps):

        if len(maps) == 0:
            maps = None
        else:
            maps = [m.reshape(m.shape[-2:] + (-1,)) for m in maps]
            maps = np.concatenate(maps, axis=2)

        accepted, assigned = non_maximum_suppresion(markers, self._threshold, self._x_disc,
                                                    self._y_disc, self._weights,
                                                    maps)

        if self._return_indices:
            indices = np.array(np.where(accepted)).T
            if assigned is not None:
                assigned = assigned[indices[:, 0], indices[:, 1]]
            return indices, assigned
        else:
            return accepted, assigned
