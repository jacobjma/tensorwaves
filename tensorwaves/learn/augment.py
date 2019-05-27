from numbers import Number

import numpy as np
from scipy.ndimage import gaussian_filter


class Augmentation(object):

    def __init__(self, p=1., image_only=False):
        self._p = p
        self._image_only = image_only

    @property
    def image_only(self):
        return self._image_only


class FlipAndRotate90(Augmentation):

    def __init__(self, p=1.):
        Augmentation.__init__(self, p=p, image_only=False)

    def apply(self, image, atoms):
        if np.random.random() < .5:
            image = np.fliplr(image)
            positions = atoms.get_positions()
            positions[:, 1] = atoms.get_cell()[1, 1] - positions[:, 1]
            atoms.set_positions(positions)

        if np.random.random() < .5:
            image = np.flipud(image)
            positions = atoms.get_positions()
            positions[:, 0] = atoms.get_cell()[0, 0] - positions[:, 0]
            atoms.set_positions(positions)

        if np.random.random() < .5:
            image = np.rot90(image)
            center = np.diag(atoms.get_cell()) / 2
            atoms.rotate(90, 'z', center=center)

        atoms.wrap(pbc=True)
        return image, atoms


class Roll(Augmentation):

    def __init__(self, p=1.):
        Augmentation.__init__(self, p=p, image_only=False)

    def apply(self, image, atoms):
        roll = np.random.randint(0, image.shape[0])
        image = np.roll(image, roll, axis=0)

        positions = atoms.get_positions()
        positions[:, 0] = positions[:, 0] + roll * atoms.get_cell()[0, 0] / image.shape[0]
        atoms.set_positions(positions)

        roll = np.random.randint(0, image.shape[1])
        image = np.roll(image, roll, axis=1)
        positions[:, 1] = positions[:, 1] + roll * atoms.get_cell()[1, 1] / image.shape[1]
        atoms.set_positions(positions)

        atoms.wrap(pbc=True)
        return image, atoms


class Crop(Augmentation):

    def __init__(self, size, p=1.):
        Augmentation.__init__(self, p=p, image_only=False)

        if isinstance(size, Number):
            size = (size, size)

        self.size = size

    def apply(self, image, atoms):
        old_size = image.shape[:2]

        shift_x = np.random.randint(0, old_size[0] - self.size[0])
        shift_y = np.random.randint(0, old_size[1] - self.size[1])

        image = image[shift_x:shift_x + self.size[0], shift_y:shift_y + self.size[1]]

        sampling = np.diag(atoms.get_cell())[:2] / old_size
        positions = atoms.get_positions()
        positions[:, :2] = positions[:, :2] - np.array((shift_x, shift_y)) * sampling
        atoms.set_positions(positions)

        del atoms[atoms.get_positions()[:, 0] < 0]
        del atoms[atoms.get_positions()[:, 1] < 0]
        del atoms[atoms.get_positions()[:, 0] > self.size[0] * sampling[0]]
        del atoms[atoms.get_positions()[:, 1] > self.size[1] * sampling[1]]

        cell = atoms.get_cell()
        cell[0, 0] = self.size[0] * sampling[0]
        cell[1, 1] = self.size[1] * sampling[1]
        atoms.set_cell(cell)

        return image, atoms


class ScaleAndShift(Augmentation):

    def __init__(self, scale=1, shift=0., scale_jitter=0., shift_jitter=0., p=1):
        Augmentation.__init__(self, p=p, image_only=True)
        self.scale = scale
        self.shift = shift
        self.scale_jitter = scale_jitter
        self.shift_jitter = shift_jitter

    def apply(self, image):
        scale = self.scale + self.scale_jitter * np.random.randn()
        shift = self.shift + self.shift_jitter * np.random.randn()
        return scale * image + shift


class NormalizeRange(Augmentation):

    def __init__(self, p=1.):
        Augmentation.__init__(self, p=p, image_only=True)

    def apply(self, image):
        return (image - image.min()) / (image.max() - image.min())


class Normalize(Augmentation):

    def __init__(self, p=1.):
        Augmentation.__init__(self, p=p, image_only=True)

    def apply(self, image):
        return (image - np.mean(image)) / np.std(image)


def normalize_local(image, sigma):
    mean = gaussian_filter(image, sigma)
    image = image - mean
    image = image / np.sqrt(gaussian_filter(image ** 2, sigma))
    return image


class NormalizeLocal(Augmentation):

    def __init__(self, sigma, p=1.):
        Augmentation.__init__(self, p=p, image_only=True)
        self.sigma = sigma

    def apply_image(self, image):
        mean = gaussian_filter(image, self.sigma)
        image = image - mean
        image = image / np.sqrt(gaussian_filter(image ** 2, self.sigma))
        return image


class GaussianBlur(Augmentation):

    def __init__(self, sigma, sigma_jitter, p=1.):
        Augmentation.__init__(self, p=p, image_only=True)
        self.sigma = sigma
        self.sigma_jitter = sigma_jitter

    def apply(self, image):
        sigma = np.max((self.sigma + np.random.randn() * self.sigma_jitter, 0.))
        if sigma > 0.:
            return gaussian_filter(image, sigma)
        else:
            return image


class Gamma(Augmentation):

    def __init__(self, gamma=1, gamma_jitter=0., p=1.):
        Augmentation.__init__(self, p=p, image_only=True)
        self.gamma = gamma
        self.gamma_jitter = gamma_jitter

    def apply(self, image):
        return image ** max(self.gamma + self.gamma_jitter * np.random.randn(), 0.)


class PoissonNoise(Augmentation):

    def __init__(self, mean_min, mean_max=None, background_min=0., background_max=None, p=1.):
        Augmentation.__init__(self, p=p, image_only=True)
        self.mean_min = mean_min
        self.mean_max = mean_max
        self.background_min = background_min
        self.background_max = background_max

    def apply(self, image):
        if self.background_max is None:
            background = self.background_min
        else:
            background = np.random.uniform(self.background_min, self.background_max)

        image = image - image.min() + background
        image = image / image.sum() * np.prod(image.shape[:2])

        if self.mean_max is None:
            mean = self.mean_min
        else:
            mean = np.random.uniform(self.mean_min, self.mean_max)

        image = np.random.poisson(image * mean).astype(np.float32)

        return image


# if image.min() < 0:
#    raise RuntimeError()
# image[image < 0.] = 0.

# scale = np.max(((self.mean + np.random.randn() * self.mean_jitter), .01))
# mean = np.mean(image)
# image = scale * image / mean
# image = np.random.poisson(image).astype(np.float32)
# return image / scale * mean


# class GaussianNoise(ImageAugmentation):
#
#     def __init__(self, amount, amount_jitter=0.):
#         ImageAugmentation.__init__(self)
#         self.amount = amount
#         self.amount_jitter = amount_jitter
#
#     def apply_image(self, image):
#         amount = self.amount + np.random.randn() * self.amount_jitter
#         image = image + np.random.randn(*image.shape) * amount
#         return image
#
#
def bandpass_noise(inner, outer, n):
    k = np.fft.fftfreq(n)
    inner = inner / n
    outer = outer / n
    mask = (k > inner) & (k < outer)
    noise = np.fft.ifft(mask * np.exp(-1.j * 2 * np.pi * np.random.rand(*k.shape)))
    noise = (noise.real + noise.imag) / 2
    return noise / np.std(noise)


def bandpass_noise_2d(inner, outer, shape):
    kx, ky = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]))
    k = np.sqrt(kx ** 2 + ky ** 2)
    inner = inner / np.max(shape)
    outer = outer / np.max(shape)
    mask = (k > inner) & (k < outer)
    noise = np.fft.ifft2(mask * np.exp(-1.j * 2 * np.pi * np.random.rand(*k.shape)))
    noise = (noise.real + noise.imag) / 2
    return noise / np.std(noise)


class ScanNoise(Augmentation):

    def __init__(self, scale, amount, fine_amount=None, p=1.):

        Augmentation.__init__(self, p=p, image_only=True)
        self.scale = scale
        self.amount = amount
        if fine_amount is None:
            self.fine_amount = self.amount / 2

    def apply(self, image):
        if self._p > np.random.rand():
            fine_scale = np.max(image.shape[:-1])

            n = ((self.amount * bandpass_noise(0, self.scale, image.shape[1]) +
                  self.amount / 2 * bandpass_noise(0, fine_scale, image.shape[1]))).astype(int)

            def strided_indexing_roll(a, r):
                from skimage.util.shape import view_as_windows
                a_ext = np.concatenate((a, a[:, :-1]), axis=1)
                n = a.shape[1]
                return view_as_windows(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]

            for i in range(image.shape[-1]):
                image[..., i] = strided_indexing_roll(image[..., i], n)

            return image
        else:
            return image
#
# class Dirt(ImageAugmentation):
#
#     def __init__(self, scale, fraction, noise_scale=None, amount=1, amount_jitter=0):
#         ImageAugmentation.__init__(self)
#         self.scale = scale
#         self.fraction = fraction
#         if noise_scale is None:
#             self.noise_scale = scale * 10
#         else:
#             self.noise_scale = noise_scale
#
#         self.amount = amount
#         self.amount_jitter = amount_jitter
#
#     def _get_mask(self, shape):
#         mask = bandpass_noise_2d(0, self.scale, shape)
#         mask = (mask - mask.min()) / (mask.max() - mask.min())
#         mask[mask > self.fraction] = self.fraction
#         # mask = mask > self.fraction
#         return (mask - mask.min()) / (mask.max() - mask.min())
#
#     def _get_noise(self, shape):
#         noise = bandpass_noise_2d(0, self.noise_scale, shape)
#         return (noise - noise.min()) / (noise.max() - noise.min())
#
#     def apply_image(self, image):
#         mask = self._get_mask(image.shape[:-1])
#         noise = self._get_noise(image.shape[:-1])
#
#         amount = np.max((0, self.amount + np.random.randn() * self.amount_jitter))
#
#         image += noise[..., None] * (1 - mask[..., None]) / noise.std() * amount
#
#         # image *= std
#
#         # label[..., 1:] *= mask[..., None] ** 5
#         # label[..., 0] = 1 - np.sum(label[..., 1:], axis=2)
#
#         return image
