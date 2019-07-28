import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist

from tensorwaves.learn.structures import SuperCell
from tensorwaves.learn.utils import generate_indices


def cluster_and_classify(positions, atomic_numbers, fingerprints=None, distance=1., return_clusters=False,
                         assign_unidentified=True):
    assert positions.shape[1] == 2
    assert len(positions) == len(atomic_numbers)

    if fingerprints is None:
        fingerprints = []

    cluster_ids = fcluster(linkage(positions), distance, criterion='distance')

    class_ids = np.zeros(len(positions), dtype=np.int32)
    class_ids[:] = -1

    for i, cluster in enumerate(np.unique(cluster_ids)):
        cluster = np.where(cluster_ids == cluster)[0]
        assignment = -1

        for j, fingerprint in enumerate(fingerprints):

            if len(cluster) == len(fingerprint):

                if np.all(fingerprint == atomic_numbers[cluster]):
                    assignment = j
                    break

        if assignment > -1:
            class_ids[cluster] = assignment

        if assign_unidentified & (assignment == -1):
            fingerprints += [atomic_numbers[cluster]]
            class_ids[cluster] = len(fingerprints) - 1

    if return_clusters:
        cluster_positions, cluster_class_ids = get_cluster_positions(positions, cluster_ids, class_ids)
        return cluster_positions, cluster_class_ids, cluster_ids, class_ids, fingerprints
    else:
        return cluster_ids, class_ids, fingerprints


def get_cluster_positions(atomic_positions, cluster_ids, class_ids):
    positions = []
    cluster_class_ids = []
    for label in range(1, cluster_ids.max() + 1):
        positions.append(np.mean(atomic_positions[cluster_ids == label], axis=0))
        cluster_class_ids.append(class_ids[cluster_ids == label][0])

    return np.array(positions)[:, :2], cluster_class_ids


def gaussian_marker_label(positions, shape, width, periodic=False):
    if isinstance(shape, int):
        shape = (shape, shape)

    margin = np.int(4 * width)
    x, y = np.mgrid[0:shape[0] + 2 * margin, 0:shape[1] + 2 * margin]
    markers = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin))

    for position in positions:
        x_lim_min = np.round(position[0]).astype(int)
        x_lim_max = np.round(position[0] + 3 * margin + 1).astype(int)
        y_lim_min = np.round(position[1]).astype(int)
        y_lim_max = np.round(position[1] + 3 * margin + 1).astype(int)

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


def labels_to_masks(labels, n_classes):
    masks = np.zeros((np.prod(labels.shape),) + (n_classes,), dtype=bool)

    for i, indices in enumerate(generate_indices(labels)):
        masks[indices, i] = True

    return masks.reshape(labels.shape + (-1,))


class Label(SuperCell):

    def __init__(self, positions, class_ids, cell=None, n_classes=None, marker_width=.5):
        self._positions = np.array(positions, np.float)

        self._n_classes = n_classes
        self._marker_width = marker_width
        arrays = {'class_ids': np.array(class_ids)}

        super().__init__(positions=positions, cell=cell, arrays=arrays)

    @property
    def class_ids(self):
        return self._arrays['class_ids']

    def scale(self, scale):

        self.positions[:] = self.positions * scale
        self.cell[:] = self.cell * scale
        self._marker_width = self._marker_width * scale

    def voronoi_masks(self, shape):
        if isinstance(shape, int):
            shape = (shape, shape)

        positions = self.positions / np.diag(self.cell) * np.array(shape)
        labels = voronoi_label(positions, self.class_ids, shape)
        masks = labels_to_masks(labels, self._n_classes)
        return masks

    def gaussian_markers(self, shape):
        if isinstance(shape, int):
            shape = (shape, shape)
        positions = self.positions / np.diag(self.cell) * np.array(shape)
        width = self._marker_width / self.cell[0, 0] * shape[0]
        return gaussian_marker_label(positions, shape, width)

    def as_images(self, shape):
        return self.gaussian_markers(shape), self.voronoi_masks(shape)

    def copy(self):
        return self.__class__(positions=self.positions.copy(), class_ids=self.class_ids.copy(), cell=self.cell.copy(),
                              n_classes=self._n_classes, marker_width=self._marker_width)


def create_label(atoms, fingerprints, n_classes, distance=.1, marker_width=.5):
    positions = atoms.get_positions()[:, :2]
    atomic_numbers = atoms.get_atomic_numbers()

    cluster_positions, class_ids, _, _, fingerprints = cluster_and_classify(positions, atomic_numbers, fingerprints,
                                                                            distance=distance, return_clusters=True,
                                                                            assign_unidentified=False)

    box = np.diag(atoms.get_cell())[:2]

    return Label(cluster_positions, class_ids, box, n_classes, marker_width=marker_width)
