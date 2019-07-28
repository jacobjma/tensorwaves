import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from tensorwaves.learn.augment import Crop
from tensorwaves.learn.labels import cluster_and_classify
from tensorwaves.potentials import Potential


def gaussian_superposition(superposition, windows, distances, widths, heights, intensities):
    gaussians = heights[None, None] * np.exp(- distances[..., None] / (2 * widths[None, None] ** 2))
    gaussians = np.sum(gaussians, axis=-1)

    for i in range(gaussians.shape[0]):
        superposition[windows[i]] += intensities[i] * gaussians[i]

    return superposition


class GaussianSuperposition(object):

    def __init__(self, shape, margin, positions):
        self._shape = shape
        self._margin = margin
        x, y = np.mgrid[0:shape[0] + 2 * margin, 0:shape[1] + 2 * margin]

        self._windows = np.zeros((len(positions), (2 * margin + 1) ** 2, 2), dtype=np.int)

        for i, position in enumerate(positions):
            x_lim_min = np.round(position[0]).astype(int)
            x_lim_max = np.round(position[0] + 2 * margin + 1).astype(int)
            y_lim_min = np.round(position[1]).astype(int)
            y_lim_max = np.round(position[1] + 2 * margin + 1).astype(int)

            self._windows[i, ..., 0] = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max].ravel()
            self._windows[i, ..., 1] = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max].ravel()

        self._distances = np.sum((self._windows - (positions[:, None] + self._margin)) ** 2, axis=-1)

        self._windows = self._windows[..., 0] * (self._shape[1] + 2 * self._margin) + self._windows[..., 1]

        self._superposition = np.zeros((self._shape[0] + 2 * self._margin) * (self._shape[1] + 2 * self._margin))

    def make_image(self, widths, heights, intensities=None):

        if intensities is None:
            intensities = np.ones(len(self._windows))

        self._superposition[:] = 0.
        image = gaussian_superposition(self._superposition, self._windows, self._distances, widths, heights,
                                       intensities)
        image = image.reshape((self._shape[0] + 2 * self._margin, self._shape[1] + 2 * self._margin))

        image = image[self._margin:-self._margin, self._margin:-self._margin]
        return image


def gaussian(x, a, b):
    return a * np.exp(-x ** 2 / (2 * b ** 2))


def fit_gaussians(data, num_gaussians):
    x = np.linspace(0, len(data) - 1, len(data))

    def residual(p):
        p = p.reshape((2, num_gaussians)).T
        p = np.abs(p)
        return np.mean((np.sum(gaussian(x[:, None], p[:, 0], p[:, 1]), axis=1) - data) ** 2)

    amplitude = data.max()
    width = np.argmin(np.abs(data - amplitude / 2))
    p0 = np.concatenate((np.linspace(amplitude / 4, amplitude, num_gaussians),
                         np.linspace(width / 2, 2 * width, num_gaussians)))

    m = minimize(residual, p0, method='nelder-mead', options={'xtol': 1e-8, 'disp': False, 'maxiter': 40000})

    p = m.x
    p = p.reshape((2, num_gaussians)).T
    p = np.abs(p)
    return p


def gaussian_atomic_columns(positions, labels, cell, shape, parameters):
    if isinstance(shape, int):
        shape = (shape, shape)

    sampling = cell / shape

    margin = np.zeros(2)
    for p in parameters.values():

        x = np.linspace(0, 40, 500)
        f = np.sum(gaussian(x, p[:, 0, None], p[:, 1, None]), axis=0)

        new_margin = x[np.argmax(f < f[0] / 5e3)] / sampling

        if new_margin[0] > margin[0]:
            margin = new_margin

    margin = margin.astype(np.int)

    xi, yi = np.mgrid[0:shape[0] + 2 * margin[0], 0:shape[1] + 2 * margin[1]]

    x = np.linspace(0, cell[0] + 2 * margin[0] * sampling[0], shape[0] + 2 * margin[0], endpoint=False)
    y = np.linspace(0, cell[1] + 2 * margin[1] * sampling[1], shape[1] + 2 * margin[1], endpoint=False)
    x, y = np.meshgrid(x, y, indexing='ij')

    image = np.zeros((shape[0] + 2 * margin[0], shape[1] + 2 * margin[1]))
    for position, pixel_position, class_id in zip(positions, positions / sampling, labels):
        x_lim_min = np.round(pixel_position[0]).astype(int)
        x_lim_max = np.round(pixel_position[0] + 2 * margin[0] + 1).astype(int)
        y_lim_min = np.round(pixel_position[1]).astype(int)
        y_lim_max = np.round(pixel_position[1] + 2 * margin[1] + 1).astype(int)

        x_window = x[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
        y_window = y[x_lim_min: x_lim_max, y_lim_min: y_lim_max]

        distances = cdist(position[None] + margin * sampling, np.array([x_window.ravel(), y_window.ravel()]).T)

        tmp = np.sum(gaussian(distances, parameters[class_id][:, 0, None], parameters[class_id][:, 1, None]), axis=0)

        xi_window = xi[x_lim_min: x_lim_max, y_lim_min: y_lim_max]
        yi_window = yi[x_lim_min: x_lim_max, y_lim_min: y_lim_max]

        image[xi_window, yi_window] += tmp.reshape(x_window.shape)

    image[margin[0]:2 * margin[0]] += image[-margin[0]:]
    image[-2 * margin[0]:-margin[0]] += image[:margin[0]]
    image[:, margin[1]:2 * margin[1]] += image[:, -margin[1]:]
    image[:, -2 * margin[1]:-margin[1]] += image[:, :margin[1]]

    markers = image[margin[0]:-margin[0], margin[1]:-margin[1]]

    return markers


def independent_column_model(atoms, probe, detector, sampling):
    positions = atoms.get_positions()[:, :2]

    atomic_numbers = atoms.get_atomic_numbers()

    cluster_positions, cluster_class_ids, cluster_ids, class_ids, _ = cluster_and_classify(positions, atomic_numbers,
                                                                                           return_clusters=True,
                                                                                           distance=1e-6)
    parameters = {}
    for unique, index in zip(*np.unique(class_ids, return_index=True)):
        column = atoms[np.where(cluster_ids == cluster_ids[index])]

        column.center(vacuum=7.5, axis=(0, 1))
        column.center(vacuum=1, axis=2)

        cell = column.get_cell()

        potential = Potential(column, num_slices=1, method='fourier')

        scan = probe.linescan(potential=potential, max_batch=500, start=np.diag(cell)[:2] / 2, end=[0, 0],
                              sampling=.1, detectors=[detector], progress_bar=False)

        data = scan.numpy()

        background = data[-1]

        data = data - background

        parameters[unique] = fit_gaussians(data, 4)
        parameters[unique][:, 1] *= scan.sampling

    cell = np.diag(atoms.get_cell())[:2]

    shape = np.ceil(cell / sampling).astype(np.int)

    return gaussian_atomic_columns(cluster_positions, cluster_class_ids, cell, shape, parameters) + background


def data_generator(images, labels, batch_size=32, compression=1, augmentations=None, classes=True):
    if augmentations is None:
        augmentations = []

    assert 16 % compression == 0

    num_iter = len(images) // batch_size
    while True:
        for i in range(num_iter):
            if i == 0:
                indices = np.arange(len(images))
                np.random.shuffle(indices)

            batch_indices = indices[i * batch_size:(i + 1) * batch_size]

            batch_images = [images[i].copy() for i in batch_indices]
            batch_labels = [labels[i].copy() for i in batch_indices]

            if classes:
                batch_image_labels = [[None] * batch_size, [None] * batch_size]
                batch_class_weights = [None] * batch_size
            else:
                batch_image_labels = [[None] * batch_size]

            for j, k in enumerate(batch_indices):
                for augmentation in augmentations:
                    batch_images[j], batch_labels[j] = augmentation.apply(batch_images[j], batch_labels[j])

                shape = (batch_images[j].shape[0] // compression, batch_images[j].shape[1] // compression)

                batch_image_labels[0][j] = batch_labels[j].gaussian_markers(shape)[..., None].astype(np.float32)
                if classes:
                    batch_image_labels[1][j] = batch_labels[j].voronoi_masks(shape).astype(np.float32)
                    # batch_labels[j]._marker_width = batch_labels[j]._marker_width #/ 1.5
                    batch_class_weights[j] = batch_image_labels[0][
                        j]  # batch_labels[j].gaussian_markers(shape)[..., None].astype(np.float32)

            batch_image_labels[0] = np.array(batch_image_labels[0]).astype(np.float32)

            if classes:
                batch_image_labels[1] = np.array(batch_image_labels[1])
                batch_class_weights = np.array(batch_class_weights).astype(np.float32)

                yield np.array(batch_images)[..., None].astype(
                    np.float32), batch_labels, batch_image_labels, batch_class_weights

            else:
                yield np.array(batch_images)[..., None].astype(np.float32), batch_labels, batch_image_labels
