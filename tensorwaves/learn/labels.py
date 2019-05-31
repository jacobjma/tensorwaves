import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from tensorwaves.learn.utils import generate_indices
from tensorwaves.learn.structures import SuperCell


def cluster_and_classify(positions, atomic_numbers, fingerprints, distance=.1):
    for i, fingerprint in enumerate(fingerprints):
        fingerprints[i] = sorted(fingerprint)

    cluster_ids = fcluster(linkage(positions), distance, criterion='distance')

    class_ids = np.zeros(cluster_ids.max(), dtype=np.int32)
    class_ids[:] = -1

    cluster_positions = np.zeros((cluster_ids.max(), 2), dtype=np.float)

    for i, cluster in enumerate(np.unique(cluster_ids)):
        cluster = np.where(cluster_ids == cluster)[0]

        cluster_positions[i] = np.mean(positions[cluster], axis=0)

        try:
            n = sorted(tuple(atomic_numbers[cluster]))
            class_id = fingerprints.index(n)
            class_ids[i] = class_id

        except:
            class_ids[i] = -1

    return cluster_positions, class_ids


def gaussian_marker_label(positions, shape, width, periodic=False):
    if isinstance(shape, int):
        shape = (shape, shape)

    margin = np.int(4 * width)
    x, y = np.mgrid[0:shape[0] + 2 * margin, 0:shape[1] + 2 * margin]
    markers = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin))

    for position in positions:
        x_lim_min = np.round(position[0]).astype(int)
        x_lim_max = np.round(position[0] + 2 * margin + 1).astype(int)
        y_lim_min = np.round(position[1]).astype(int)
        y_lim_max = np.round(position[1] + 2 * margin + 1).astype(int)

        x_window = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
        y_window = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max]

        distances = cdist(position[None] + margin, np.array([x_window.ravel(), y_window.ravel()]).T)

        gaussian = np.exp(- distances ** 2 / (2 * width ** 2))

        markers[x_window, y_window] += gaussian.reshape(x_window.shape)

    if periodic:
        markers[margin:2 * margin] += markers[-margin:]
        markers[-2 * margin:-margin] += markers[:margin]
        markers[:, margin:2 * margin] += markers[:, -margin:]
        markers[:, -2 * margin:-margin] += markers[:, :margin]

    markers = markers[margin:-margin, margin:-margin]

    return markers


def closest_position_label(positions, shape):
    from skimage.morphology import watershed

    markers = np.zeros(shape, dtype=np.int)
    indices = (positions).astype(int)
    markers[indices[:, 0], indices[:, 1]] = range(1, len(positions) + 1)
    labels = watershed(np.zeros_like(markers), markers, compactness=1000)

    x = np.zeros(labels.shape, dtype=np.float).flatten()
    y = np.zeros(labels.shape, dtype=np.float).flatten()

    for i, indices in enumerate(generate_indices(labels)):
        x[indices] = positions[i, 0]
        y[indices] = positions[i, 1]

    return np.stack([x.reshape(labels.shape), y.reshape(labels.shape)], -1).reshape(shape + (2,))


def voronoi_label(positions, class_ids, shape):
    from skimage.morphology import watershed

    markers = np.zeros(shape, dtype=np.int)
    indices = (positions).astype(int)
    markers[indices[:, 0], indices[:, 1]] = 1 + class_ids  # range(1, len(positions) + 1)
    labels = watershed(np.zeros_like(markers), markers, compactness=1000)

    return labels


def labels_to_masks(labels):
    masks = np.zeros((np.prod(labels.shape),) + (labels.max(),), dtype=bool)

    for i, indices in enumerate(generate_indices(labels)):
        masks[indices, i] = True

    return masks.reshape(labels.shape + (-1,))


class Label(SuperCell):

    def __init__(self, positions, class_ids, cell=None):
        self._positions = np.array(positions, np.float)

        arrays = {'class_ids': np.array(class_ids)}

        super().__init__(positions=positions, cell=cell, arrays=arrays)

    @property
    def class_ids(self):
        return self._arrays['class_ids']

    def voronoi_masks(self, shape):
        if isinstance(shape, int):
            shape = (shape, shape)

        positions = self.positions / np.diag(self.cell) * np.array(shape)
        labels = voronoi_label(positions, self.class_ids, shape)
        masks = labels_to_masks(labels)
        return masks

    def gaussian_markers(self, shape, width):
        if isinstance(shape, int):
            shape = (shape, shape)
        positions = self.positions / np.diag(self.cell) * np.array(shape)
        width = width / self.cell[0, 0] * shape[0]
        return gaussian_marker_label(positions, shape, width)

    def copy(self):
        return self.__class__(positions=self.positions.copy(), class_ids=self.class_ids.copy(), cell=self.cell.copy())


def create_label(atoms, fingerprints, distance=.1):
    positions = atoms.get_positions()[:, :2]
    atomic_numbers = atoms.get_atomic_numbers()

    cluster_positions, class_ids = cluster_and_classify(positions, atomic_numbers, fingerprints, distance=distance)

    box = np.diag(atoms.get_cell())[:2]

    return Label(cluster_positions, class_ids, box)
