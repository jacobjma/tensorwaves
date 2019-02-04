import os

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist

from tensorwaves.bases import HasGrid, GridProperty
from tensorwaves.potentials import Potential
from tensorwaves.scan import GridScan
from tensorflow import keras
import scipy.ndimage
from sklearn.model_selection import train_test_split


def regression_image_generator(X, Y, seed=0):
    # X = scipy.ndimage.zoom(X, (1, 2, 2, 1))
    # Y = scipy.ndimage.zoom(Y, (1, 2, 2, 1))

    def image_preproccesing_function(x):
        x = 1 + 2 * np.random.rand() + (x - x.min()) / (x.max() - x.min()) * 4 * (.2 + np.random.rand())
        x = np.random.poisson(x).astype(float)
        x += .5 * np.random.rand() * np.random.normal(0, 1, x.shape)
        x -= np.min(x)
        x = x ** (1 + .2 * 2 * (np.random.rand() - .5))
        x -= np.min(x)
        x /= np.std(x)
        #x *= 1 + .2 * 2 * (np.random.rand() - .5)
        #x += np.random.rand() * x.max() * .05
        return x

    def preprocessing(x, fliplr=False, flipud=False, rotation=0., zoom=1., rolllr=0, rollud=0):

        if fliplr:
            x = np.fliplr(x)

        if flipud:
            x = np.fliplr(x)

        if rotation > 0.:
            x = scipy.ndimage.rotate(x, rotation, reshape=False)

        if zoom != 1.:
            x_new = scipy.ndimage.zoom(x, (zoom, zoom, 1))
            x_new = x_new[:x.shape[0], :x.shape[1], :]

            x = np.zeros_like(x)
            x[:x_new.shape[0], :x_new.shape[1], :] = x_new

        if rolllr > 0:
            x = np.roll(x, rolllr, axis=0)

        if rollud > 0:
            x = np.roll(x, rollud, axis=0)

        return x

    datagen_x = keras.preprocessing.image.ImageDataGenerator()
    datagen_y = keras.preprocessing.image.ImageDataGenerator()

    train_generator_x = datagen_x.flow(X, batch_size=16, seed=seed)
    train_generator_y = datagen_y.flow(Y, batch_size=16, seed=seed)

    train_generator = zip(train_generator_x, train_generator_y)

    for X_batch, Y_batch in train_generator:

        for i in range(X_batch.shape[0]):

            X_batch[i] -= X_batch[i].min()

            if np.random.random() < 0.5:
                fliplr = True
            else:
                fliplr = False

            if np.random.random() < 0.5:
                flipud = True
            else:
                flipud = False

            if np.random.random() < 0.5:
                rotation = np.random.uniform(0., 360)
            else:
                rotation = 0.

            zoom = np.random.uniform(.8, 1.2)
            # zoom = 1
            rolllr = np.round(np.random.uniform(0, X_batch[i].shape[0])).astype(int)
            rollud = np.round(np.random.uniform(0, X_batch[i].shape[1])).astype(int)

            X_batch[i] = preprocessing(X_batch[i], fliplr=fliplr, flipud=flipud, rotation=rotation, zoom=zoom,
                                       rolllr=rolllr, rollud=rollud)
            Y_batch[i] = preprocessing(Y_batch[i], fliplr=fliplr, flipud=flipud, rotation=rotation, zoom=zoom,
                                       rolllr=rolllr, rollud=rollud)

            Y_batch[i, ..., -1] = 1 - np.sum(Y_batch[i, ..., :-1], axis=2)

            X_batch[i] = image_preproccesing_function(X_batch[i])

        yield X_batch, Y_batch


def cluster_and_classify(atoms, distance=1., fingerprints=None):
    positions = atoms.get_positions()[:, :2]

    cluster_labels = fcluster(linkage(positions), distance, criterion='distance')

    class_labels = -np.ones(len(atoms), dtype=np.int32)

    for i, cluster_label in enumerate(np.unique(cluster_labels)):
        cluster = np.where(cluster_labels == cluster_label)[0]
        for j, fingerprint in enumerate(fingerprints):
            if len(cluster) == len(fingerprint):
                if np.all(fingerprint == atoms.get_atomic_numbers()[cluster]):
                    class_labels[cluster] = j
                    break

    return cluster_labels, class_labels


def create_gt(cluster_centers, gpts, extent, width, max_label=None):
    sampling = extent / gpts

    if max_label is None:
        max_label = len(cluster_centers)

    x, y = np.mgrid[0:gpts[0], 0:gpts[1]]
    gt = np.zeros(tuple(gpts) + (max_label + 1,))

    for i, positions in enumerate(cluster_centers.values()):
        positions /= sampling

        for position in positions:
            rounded_position = np.round(position).astype(int)

            x_lim_min = np.max((rounded_position[0] - width * 4, 0))
            x_lim_max = np.min((rounded_position[0] + width * 4 + 1, gpts[0]))
            y_lim_min = np.max((rounded_position[1] - width * 4, 0))
            y_lim_max = np.min((rounded_position[1] + width * 4 + 1, gpts[1]))

            x_window = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
            y_window = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max]

            gaussian = np.exp(-cdist(position[None], np.array([x_window.ravel(), y_window.ravel()]).T) ** 2 / width)
            gt[x_window, y_window, i] += gaussian.reshape(x_window.shape)

    gt[:, :, -1] = 1 - np.sum(gt, axis=2)

    return gt


class TrainingSet(object):

    def __init__(self, examples):
        self._examples = examples

    @property
    def num_examples(self):
        return len(self._examples)

    def save(self, base_name, padding=3):
        for i, example in enumerate(self._examples):
            example.save(base_name + '_{}.npz'.format(str(i).zfill(padding)))

    def __getitem__(self, i):
        return self._examples[i]


class TrainingExample(object):

    def __init__(self, atoms, truth=None):
        self._atoms = atoms
        self._truth = truth

    @property
    def atoms(self):
        return self._atoms

    @property
    def truth(self):
        return self._truth


class TrainingSetSTEM(TrainingSet):

    def __init__(self, examples=None):
        TrainingSet.__init__(self, examples)

    def as_tensors(self):
        size = tuple(self._examples[0].grid.gpts)
        num_classes = self._examples[0].num_classes + 1

        images = np.zeros((self.num_examples,) + size + (1,))
        labels = np.zeros((self.num_examples,) + size + (num_classes,))

        for i, example in enumerate(self._examples):
            images[i] = example._image[:, :, :, None]
            labels[i] = example._truth[:, :, :]

        return images, labels

    def get_generators(self):
        X, Y = self.as_tensors()
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.15, random_state=2018)

        train_generator = regression_image_generator(X_train, Y_train)
        valid_generator = regression_image_generator(X_valid, Y_valid)

        return train_generator, valid_generator

    @property
    def height(self):
        return self._examples[0].grid.gpts[0]

    @property
    def width(self):
        return self._examples[0].grid.gpts[1]

    def load(self, prefix):
        folder = prefix.split('/')[:-1]
        if len(folder) == 0:
            prefix += '/'
            folder = prefix.split('/')[:-1]

        files = os.listdir('/'.join(folder))
        base_name = prefix.split('/')[-1]
        files = ['/'.join(folder + [file]) for file in files if file[:len(base_name)] == base_name]

        self._examples = []
        for file in files:
            self._examples.append(load_training_example_stem(file))

    def create_ground_truth(self, distance=None, fingerprints=None, gaussian_width=None, max_label=None):

        for example in self._examples:
            example.create_ground_truth(distance=distance, fingerprints=fingerprints, gaussian_width=gaussian_width,
                                        max_label=max_label)


class TrainingExampleSTEM(HasGrid):

    def __init__(self, atoms, num_positions=None, sampling=None):
        self._atoms = atoms

        extent = GridProperty(lambda: np.diag(self._atoms.get_cell())[:2], dtype=np.float32, locked=True)

        HasGrid.__init__(self, extent=extent, gpts=num_positions, sampling=sampling)

        self._cluster_labels = None
        self._class_labels = None
        self._truth = None
        self._image = None

    @property
    def num_classes(self):
        return np.max(self._class_labels) + 1

    def scale_image(self, num_positions):

        self.grid.gpts = num_positions

        # for i, image in enumerate(self._images):
        self._image = scipy.ndimage.zoom(self._image, (1, 2, 2))

    def classify(self, distance=None, fingerprints=None):

        if distance is None:
            # TODO : Implement this
            distance = 0.

        if fingerprints is None:
            # TODO : Implement this
            raise NotImplementedError()

        cluster_labels, class_labels = cluster_and_classify(self._atoms, distance, fingerprints)

        self._cluster_labels = cluster_labels
        self._class_labels = class_labels

    def create_ground_truth(self, distance=None, fingerprints=None, gaussian_width=None, max_label=None):

        self.classify(distance, fingerprints)

        # print(self._class_labels)

        self._truth = self._calculate_truth(self._cluster_labels, self._class_labels, gaussian_width,
                                            max_label=max_label)

    def create_image(self, prism, detector, tracker):
        prism.grid.extent = self.grid.extent

        potential = Potential(self._atoms, sampling=prism.grid.sampling)

        S = prism.get_scattering_matrix()

        S.multislice(potential, in_place=True, tracker=tracker)

        scan = GridScan(scanable=S, detectors=detector, num_positions=self.grid.gpts)

        scan.scan(tracker=tracker)

        self._image = scan.read_detector().numpy()

    def get_centers(self):
        cluster_centers = []

        for i, class_label in enumerate(np.unique(self._class_labels)):
            class_indices = np.where(self._class_labels == class_label)[0]

            class_clusters = [np.where(self._cluster_labels[class_indices] == unique)[0] for unique in
                              np.unique(self._cluster_labels[class_indices])]

            cluster_centers += [np.mean(self._atoms.get_positions()[class_indices][class_clusters, :2], axis=1)]

        return cluster_centers

    def _calculate_truth(self, cluster_labels, class_labels, gaussian_width, max_label=None):

        if max_label is None:
            max_label = np.max(class_labels)

        x, y = np.mgrid[0:self.grid.gpts[0], 0:self.grid.gpts[1]]
        truth = np.zeros(tuple(self.grid.gpts) + (max_label + 2,))

        for cluster_label in cluster_labels:
            cluster_indices = np.where(cluster_label == cluster_labels)[0]
            position = np.mean(self._atoms.get_positions()[cluster_indices, :2], axis=0) / self.grid.sampling

            rounded_position = np.round(position).astype(int)

            x_lim_min = np.max((rounded_position[0] - gaussian_width * 4, 0))
            x_lim_max = np.min((rounded_position[0] + gaussian_width * 4 + 1, self.grid.gpts[0]))
            y_lim_min = np.max((rounded_position[1] - gaussian_width * 4, 0))
            y_lim_max = np.min((rounded_position[1] + gaussian_width * 4 + 1, self.grid.gpts[1]))

            x_window = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
            y_window = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max]

            gaussian = np.exp(
                -cdist(position[None], np.array([x_window.ravel(), y_window.ravel()]).T) ** 2 / gaussian_width)
            truth[x_window, y_window, class_labels[cluster_indices[0]]] += gaussian.reshape(x_window.shape)

        truth[:, :, -1] = 1 - np.sum(truth, axis=2)

        return truth

    def show_classification(self):
        import matplotlib.pyplot as plt
        cluster_centers = self.get_centers()

        for i, centers in enumerate(cluster_centers):
            plt.scatter(*centers.T, label=i)

        plt.axis('equal')
        plt.legend()

    def show_truth(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(self._truth[:, :, i].T, origin='lower')

    def show_image(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(self._image[0].T, origin='lower')

    def save(self, outfile):

        saved = {'positions': self._atoms.get_positions(),
                 'atomic_numbers': self._atoms.get_atomic_numbers(),
                 'cell': self._atoms.get_cell(),
                 'cluster_labels': self._cluster_labels,
                 'class_labels': self._class_labels,
                 'truth': self._truth,
                 'image': self._image}

        np.savez(outfile, **saved)


def load_training_example_stem(file):
    from ase import Atoms

    loaded = np.load(file)

    atoms = Atoms(positions=loaded['positions'], numbers=loaded['atomic_numbers'], cell=loaded['cell'])

    example = TrainingExampleSTEM(atoms=atoms, num_positions=loaded['image'].shape[1:])

    example._cluster_labels = loaded['cluster_labels']
    example._class_labels = loaded['class_labels']
    example._truth = loaded['truth']
    example._image = loaded['image']

    return example
