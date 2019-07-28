import numpy as np
import tensorflow as tf
from scipy import ndimage
from scipy.signal import find_peaks


def _unroll(f, inner, outer, nbins_angular=32, nbins_radial=None):
    if nbins_radial is None:
        nbins_radial = outer - inner

    shape = f.shape

    sx, sy = shape
    X, Y = np.ogrid[0:sx, 0:sy]
    r = np.hypot(X - sx / 2, Y - sy / 2)
    radial_bins = -np.ones(shape, dtype=int)
    valid = (r > inner) & (r < outer)
    radial_bins[valid] = nbins_radial * (r[valid] - inner) / (outer - inner)

    angles = np.arctan2(X - sx // 2, Y - sy // 2) + np.pi
    angular_bins = np.floor(nbins_angular * (angles / (2 * np.pi)))
    angular_bins = np.clip(angular_bins, 0, nbins_angular - 1).astype(np.int)

    unrolled = np.zeros((nbins_radial, nbins_angular))
    for i in range(nbins_radial):
        j = radial_bins == i
        unrolled[i] = ndimage.mean(f[j], angular_bins[j], range(0, nbins_angular))

    return unrolled


def estimate_scale(image, d, lattice, nbins_angular=32):
    if lattice == 'rectangular':
        c = 1.

    elif lattice == 'hexagonal':
        c = np.sqrt(3) / 2

    else:
        raise RuntimeError('choose hexagonal or rectangular lattice')

    if image.shape[0] != image.shape[1]:
        raise RuntimeError('requires square image')

    n = image.shape[0]
    F = np.fft.fftshift(np.abs(np.fft.fft2(image)) ** 2)
    unrolled = _unroll(F, 1, n // 2, nbins_angular=nbins_angular)
    f = np.sum(unrolled, axis=1)

    peaks = find_peaks(f)[0]

    max_peak = peaks[np.argmax(f[peaks])]

    return (max_peak + 1) * d / float(n) * c


class ScaleEstimator(object):

    def estimate_scale(self, image):
        raise NotImplementedError()


class FourierScaleEstimator(ScaleEstimator):

    def __init__(self, d, symmetry):
        self.d = d
        self.symmetry = symmetry

    def estimate_scale(self, image):
        if len(image.shape) == 3:
            image = image[..., 0]

        return estimate_scale(image, self.d, self.symmetry)


def pad_to_closest_multiple(images, m):
    target_height = int(np.ceil(images.shape[1] / m) * m)
    target_width = int(np.ceil(images.shape[2] / m) * m)
    images = tf.image.pad_to_bounding_box(images, 0, 0, target_height, target_width)
    return images


def normalize_global(images):
    moments = tf.nn.moments(images, axes=[1, 2], keepdims=True)
    return (images - moments[0]) / tf.sqrt(moments[1])


def gaussian_kernel(sigma, truncate=None, dtype=tf.float32):
    if truncate is None:
        truncate = tf.math.ceil(3 * tf.cast(sigma, dtype))
    x = tf.cast(tf.range(-truncate, truncate + 1), dtype)
    return 1. / (np.sqrt(2. * np.pi) * sigma) * tf.exp(-x ** 2 / (2 * sigma ** 2))


def gaussian_filter(image, sigma, truncate=None):
    kernel = gaussian_kernel(sigma, truncate)
    image = tf.nn.conv2d(image, kernel[..., None, None, None], strides=[1, 1, 1, 1], padding='SAME')
    image = tf.nn.conv2d(image, kernel[None, ..., None, None], strides=[1, 1, 1, 1], padding='SAME')
    return image


def normalize_local(images, sigma):
    truncate = tf.cast(tf.math.ceil(3 * tf.cast(sigma, tf.float32)), tf.int32)
    images = tf.pad(images, [[0, 0], [truncate, truncate], [truncate, truncate], [0, 0]], 'REFLECT')
    mean = gaussian_filter(images, sigma, truncate)
    images = images - mean
    images = images / tf.sqrt(gaussian_filter(images ** 2, sigma, truncate))
    return images[:, truncate:-truncate, truncate:-truncate, :]


def rescale(images, scale):
    new_shape = np.round(np.array(images.shape[1:-1]) * scale).astype(int)
    return tf.image.resize(images, new_shape)
