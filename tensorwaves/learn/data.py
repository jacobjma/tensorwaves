import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist


def cluster_and_classify(atoms=None, atomic_positions=None, atomic_numbers=None, distance=1., fingerprints=None,
                         return_clusters=False, assign_unidentified=-1):
    if atoms is not None:
        atomic_positions = atoms.get_positions()[:, :2]
        atomic_numbers = atoms.get_atomic_numbers()
    elif (atomic_positions is None) | (atomic_numbers is None):
        raise RuntimeError()

    if fingerprints is None:
        fingerprints = ()

    cluster_ids = fcluster(linkage(atomic_positions), distance, criterion='distance')

    class_ids = np.zeros(len(atoms), dtype=np.int32)
    class_ids[:] = assign_unidentified

    for i, cluster in enumerate(np.unique(cluster_ids)):
        cluster = np.where(cluster_ids == cluster)[0]

        for j, fingerprint in enumerate(fingerprints):

            if len(cluster) == len(fingerprint):

                if np.all(fingerprint == atomic_numbers[cluster]):
                    class_ids[cluster] = j
                    break

    if return_clusters:
        return cluster_positions(atomic_positions, cluster_ids, class_ids)
    else:
        return cluster_ids, class_ids


def cluster_positions(atomic_positions, cluster_ids, class_ids):
    positions = []
    cluster_class_ids = []
    for label in range(1, cluster_ids.max() + 1):
        positions.append(np.mean(atomic_positions[cluster_ids == label], axis=0))
        cluster_class_ids.append(class_ids[cluster_ids == label][0])

    return np.array(positions)[:, :2], cluster_class_ids


def gaussian_marker_label(atoms, shape, width, periodic=False):
    positions = atoms.get_positions()[:, :2]

    positions = positions / np.diag(atoms.get_cell())[:2] * shape

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


def generate_indices(labels):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(1, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


def voronoi_label(atoms, shape):
    from skimage.morphology import watershed

    positions = atoms.get_positions()[:, :2]

    positions = positions / np.diag(atoms.get_cell())[:2] * shape

    markers = np.zeros(shape, dtype=np.int)
    indices = (positions).astype(int)
    markers[indices[:, 0], indices[:, 1]] = range(1, len(positions) + 1)
    labels = watershed(np.zeros_like(markers), markers, compactness=1000)

    return labels


def voronoi_masks(atoms, shape):
    labels = voronoi_label(atoms, shape)

    masks = np.zeros((shape[0] * shape[1],) + (len(atoms),), dtype=bool)

    for i, indices in enumerate(generate_indices(labels)):
        masks[indices, i] = True

    return masks.reshape(shape + (-1,))


def closest_position_label(atoms, shape):
    from skimage.morphology import watershed

    positions = atoms.get_positions()[:, :2]

    positions = positions / np.diag(atoms.get_cell())[:2] * shape

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


def apply_augmentations(images, atoms, augmentations):
    for i in range(len(images)):
        for augmentation in augmentations:
            if augmentation.image_only:
                images[i] = augmentation.apply(images[i])
            else:
                images[i], atoms[i] = augmentation.apply(images[i], atoms[i])

    return images, atoms


def data_generator(images, atoms, label_func, batch_size=32, shuffle=True, augmentations=None):
    if augmentations is None:
        augmentations = []

    num_iter = len(images) // batch_size
    assert images.shape[1] == images.shape[2]

    while True:
        for i in range(num_iter):
            if i == 0:
                indices = np.arange(len(images))
                np.random.shuffle(indices)

            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_images = [images[i].copy() for i in batch_indices]
            batch_atoms = [atoms[i].copy() for i in batch_indices]

            for j in range(batch_size):
                for augmentation in augmentations:
                    if augmentation.image_only:
                        batch_images[j] = augmentation.apply(batch_images[j])
                    else:
                        batch_images[j], batch_atoms[j] = augmentation.apply(batch_images[j], batch_atoms[j])

            batch_images = np.stack(batch_images)

            size = batch_images.shape[1:-1]
            #displacements = np.zeros((batch_size, size // 4, size // 4, 2), dtype=np.float32)
            #markers = np.zeros((batch_size, size // 4, size // 4, 1), dtype=np.float32)
            #instances = np.zeros((batch_size, size // 4, size // 4, 1), dtype=np.int)

            #neighbors = []
            #neighbor_weights = []
            # weights = np.zeros((batch_size, size // 4, size // 4, 1), dtype=np.float32)

            batch_labels = label_func(batch_atoms, size)

            yield batch_images, batch_labels
