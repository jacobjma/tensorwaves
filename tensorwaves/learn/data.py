import os

import numpy as np
import tensorflow as tf
from ase.io import read
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
from skimage.transform import rescale

from tensorwaves.bases import HasGrid, GridProperty
from sklearn.model_selection import train_test_split


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


def gaussian_marker_label(positions, cluster_class_ids, shape, width, depth, include_null=True):
    margin = np.int(4 * width)
    x, y = np.mgrid[0:shape[0] + 2 * margin, 0:shape[1] + 2 * margin]
    label = np.zeros((shape[0] + 2 * margin, shape[1] + 2 * margin) + (depth,))

    for position, cluster_class_id in zip(positions, cluster_class_ids):
        x_lim_min = np.round(position[0]).astype(int)
        x_lim_max = np.round(position[0] + 2 * margin + 1).astype(int)
        y_lim_min = np.round(position[1]).astype(int)
        y_lim_max = np.round(position[1] + 2 * margin + 1).astype(int)

        x_window = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
        y_window = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max]

        gaussian = np.exp(-cdist(position[None] + margin, np.array([x_window.ravel(),
                                                                    y_window.ravel()]).T) ** 2 / (
                                  2 * width ** 2))
        label[x_window, y_window, cluster_class_id + 1] += gaussian.reshape(x_window.shape)

    label[margin:2 * margin] += label[-margin:]
    label[-2 * margin:-margin] += label[:margin]
    label[:, margin:2 * margin] += label[:, -margin:]
    label[:, -2 * margin:-margin] += label[:, :margin]

    label = label[margin:-margin, margin:-margin]

    if include_null:
        label[:, :, 0] = 1 - np.sum(label, axis=2)

        return label
    else:
        return label[..., 1:]


class Sequence(tf.keras.utils.Sequence):
    def __init__(self, training_set, batch_size=32, shuffle=True, augmentations=None):

        self.training_set = training_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.on_epoch_end()

        image, label = training_set[0].as_tensors(augmentations=augmentations)
        self.image_shape = image.shape[1:]
        self.label_shape = label.shape[1:]

    def __len__(self):
        return int(np.floor(len(self.training_set) / self.batch_size))

    def __getitem__(self, i):
        # Generate indexes of the batch
        batch_indices = self.indices[i * self.batch_size:(i + 1) * self.batch_size]

        if i == len(self) - 1:
            self.on_epoch_end()

        batch_images = np.zeros((self.batch_size,) + self.image_shape)
        batch_labels = np.zeros((self.batch_size,) + self.label_shape)

        for j, batch_index in enumerate(batch_indices):
            image, label = self.training_set[batch_index].as_tensors(self.augmentations)
            batch_images[j] = image
            batch_labels[j] = label

        return (batch_images, batch_labels), (batch_labels, batch_labels)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.training_set))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


class TrainingSet(object):

    def __init__(self, examples=None):
        self._examples = examples

    def __getitem__(self, i):
        return self._examples[i]

    def __len__(self):
        return len(self._examples)

    @property
    def num_examples(self):
        return len(self._examples)

    @property
    def image_shape(self):
        return self._examples[0]._image.shape

    @property
    def label_shape(self):
        return tuple(self._examples[0].gpts) + (3,)

    def generator(self, batch_size=32, shuffle=True, augmentations=None):
        def g():
            while True:
                sequence = Sequence(self, batch_size=batch_size, shuffle=shuffle, augmentations=augmentations)
                for i in range(len(sequence)):
                    yield sequence[i]

        return g

    @property
    def height(self):
        return self._examples[0].gpts[0]

    @property
    def width(self):
        return self._examples[0].gpts[1]

    def split(self, test_size):
        train_examples, test_examples = train_test_split(self._examples, test_size=test_size)
        return TrainingSet(train_examples), TrainingSet(test_examples)

    def apply_to_all(self, func, *args):
        for example in self._examples:
            getattr(example, func)(*args)

    def cluster_and_classify(self, distance, fingerprints=None, assign_unidentified=-1):
        self.apply_to_all('cluster_and_classify', distance, fingerprints, assign_unidentified)

    def create_labels(self, gaussian_width=.25, depth=None, include_null=True):
        self.apply_to_all('create_label', gaussian_width, depth, include_null)

    def resample(self, sampling):
        self.apply_to_all('resample', sampling)


def load_training_set(image_dir, atoms_dir=None):
    if atoms_dir is None:
        atoms_dir = image_dir

    image_files = sorted([file for file in os.listdir(atoms_dir) if file[:5] == 'image'])
    atoms_files = sorted([file for file in os.listdir(atoms_dir) if file[:5] == 'atoms'])

    examples = []
    for image_file, atoms_file in zip(image_files, atoms_files):
        atoms = read(os.path.join(atoms_dir, atoms_file))
        image = np.load(os.path.join(image_dir, image_file))[0]
        examples.append(TrainingExample(atoms, image))

    return TrainingSet(examples)


class TrainingExample(HasGrid):

    def __init__(self, atoms, image):
        self._atoms = atoms
        self._image = image.reshape(image.shape[:2] + (-1,))
        self._cluster_ids = None
        self._class_ids = None
        self._label = None

        extent = GridProperty(lambda: np.diag(self._atoms.get_cell())[:2], dtype=np.float32, locked=True)
        gpts = GridProperty(lambda: self._image.shape[:-1], dtype=np.int32, locked=True)

        HasGrid.__init__(self, extent=extent, gpts=gpts)

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label

    @property
    def num_classes(self):
        return np.max(self._class_ids) + 1

    def as_tensors(self, augmentations=None):
        image, label = self._image.copy(), self._label.copy()

        if augmentations:
            for augmentation in augmentations:

                try:
                    image, label = augmentation.apply_image_and_label(image, label)
                except AttributeError:
                    pass

                try:
                    image = augmentation.apply_image(image)
                except AttributeError:
                    pass

        return image[None], label[None]

    def resample(self, sampling):
        scale = self.sampling / sampling
        self._image = rescale(self._image, scale, multichannel=True, mode='wrap', anti_aliasing=False, order=1)

    def cluster_and_classify(self, distance, fingerprints=None, assign_unidentified=-1):

        if fingerprints is None:
            # TODO : Implement this
            raise NotImplementedError()

        cluster_ids, class_ids = cluster_and_classify(self._atoms, distance=distance, fingerprints=fingerprints)

        class_ids[class_ids == -1] = assign_unidentified

        self._cluster_ids = cluster_ids
        self._class_ids = class_ids

    def create_label(self, gaussian_width=.25, depth=None, include_null=True):

        if depth is None:
            depth = self.num_classes + 1

        positions, cluster_class_ids = cluster_positions(self._atoms.get_positions(), self._cluster_ids,
                                                         self._class_ids)

        self._label = gaussian_marker_label(positions / self.sampling, cluster_class_ids, self.gpts, gaussian_width,
                                            depth,
                                            include_null=include_null)

        #
        # gaussian_width = gaussian_width / np.mean(self.sampling)
        #
        # margin = np.int(4 * gaussian_width)
        # x, y = np.mgrid[0:self.gpts[0] + 2 * margin, 0:self.gpts[1] + 2 * margin]
        # self._label = np.zeros((self.gpts[0] + 2 * margin, self.gpts[1] + 2 * margin) + (depth,))
        #
        # for position, cluster_class_id in zip(*cluster_positions(self._atoms, self._cluster_ids, self._class_ids)):
        #     position /= self.sampling
        #
        #     x_lim_min = np.round(position[0]).astype(int)
        #     x_lim_max = np.round(position[0] + 2 * margin + 1).astype(int)
        #     y_lim_min = np.round(position[1]).astype(int)
        #     y_lim_max = np.round(position[1] + 2 * margin + 1).astype(int)
        #
        #     x_window = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
        #     y_window = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
        #
        #     gaussian = np.exp(-cdist(position[None] + margin, np.array([x_window.ravel(),
        #                                                                 y_window.ravel()]).T) ** 2 / (
        #                               2 * gaussian_width ** 2))
        #     self._label[x_window, y_window, cluster_class_id + 1] += gaussian.reshape(x_window.shape)
        #
        # self._label[margin:2 * margin] += self._label[-margin:]
        # self._label[-2 * margin:-margin] += self._label[:margin]
        # self._label[:, margin:2 * margin] += self._label[:, -margin:]
        # self._label[:, -2 * margin:-margin] += self._label[:, :margin]
        #
        # self._label = self._label[margin:-margin, margin:-margin]
        #
        # self._label[:, :, 0] = 1 - np.sum(self._label, axis=2)

    def show_clusters(self):
        import matplotlib.pyplot as plt
        positions, cluster_class_ids = cluster_positions(self._atoms.get_positions(), self._cluster_ids,
                                                         self._class_ids)

        plt.imshow(self._image[..., 0].T, origin='lower', extent=[0, self.extent[0], 0, self.extent[1]], cmap='gray')

        for cluster_class_id in np.unique(cluster_class_ids):
            class_positions = positions[np.where(cluster_class_ids == cluster_class_id)]
            plt.scatter(*class_positions.T, label=cluster_class_id)

        plt.xlim([0, self.extent[0]])
        plt.ylim([0, self.extent[1]])

    def show_label(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, self.num_classes + 1, figsize=(12, 5))
        for i, ax in enumerate(axes):
            ax.imshow(self._label[:, :, i].T, origin='lower')

    def show_image(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(self._image[..., 0].T, origin='lower', extent=[0, self.extent[0], 0, self.extent[1]], cmap='gray')
