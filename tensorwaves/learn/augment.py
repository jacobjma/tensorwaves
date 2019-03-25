import numpy as np
import scipy.ndimage
from skimage.util.shape import view_as_windows
from numbers import Number
from scipy.ndimage import gaussian_filter


class Augmentation(object):

    def __init__(self):
        pass


class CommonAugmentation(Augmentation):

    def __init__(self):
        Augmentation.__init__(self)

    def apply_image_and_label(self, image, label):
        raise NotImplementedError()


class ImageAugmentation(Augmentation):

    def __init__(self):
        Augmentation.__init__(self)

    def apply_image(self, image):
        raise NotImplementedError()


class FlipAndRotate90(CommonAugmentation):

    def __init__(self):
        CommonAugmentation.__init__(self)

    def apply_image_and_label(self, image, label):

        if np.random.random() < 0.5:
            image = np.fliplr(image)
            label = np.fliplr(label)

        if np.random.random() < 0.5:
            image = np.flipud(image)
            label = np.flipud(label)

        if np.random.random() < 0.5:
            image = np.rot90(image)
            label = np.rot90(label)

        return image, label


class Roll(CommonAugmentation):

    def __init__(self):
        CommonAugmentation.__init__(self)

    def apply_image_and_label(self, image, label):
        roll = np.random.randint(0, image.shape[0])
        image = np.roll(image, roll, axis=0)
        label = np.roll(label, roll, axis=0)

        roll = np.random.randint(0, image.shape[1])
        image = np.roll(image, roll, axis=1)
        label = np.roll(label, roll, axis=1)

        return image, label


class Zoom(CommonAugmentation):

    def __init__(self, zoom=1., zoom_jitter=0.):
        CommonAugmentation.__init__(self)
        self.zoom = zoom
        self.zoom_jitter = zoom_jitter

    def _apply(self, x, zoom):
        shape = x.shape

        zoomed = scipy.ndimage.zoom(x, (zoom, zoom, 1))

        # return zoomed

        if zoom < 1.:
            new_zoomed = np.zeros(shape)
            new_zoomed[:zoomed.shape[0], :zoomed.shape[1]] = zoomed  # [:shape[0], :shape[1]]
            return new_zoomed
        else:
            return zoomed[:shape[0], :shape[1]]

    def apply_image_and_label(self, image, label):
        zoom = self.zoom + self.zoom_jitter * np.random.randn()
        return self._apply(image, zoom), self._apply(label, zoom)


class Crop(CommonAugmentation):

    def __init__(self, size):
        CommonAugmentation.__init__(self)

        if isinstance(size, Number):
            size = (size, size)

        self.size = size

    def apply_image_and_label(self, image, label):
        old_size = image.shape[:2]

        shift_x = np.random.randint(0, old_size[0] - self.size[0])
        shift_y = np.random.randint(0, old_size[1] - self.size[1])

        image = image[shift_x:shift_x + self.size[0], shift_y:shift_y + self.size[1]]
        label = label[shift_x:shift_x + self.size[0], shift_y:shift_y + self.size[1]]

        return image, label


class ScaleAndShift(ImageAugmentation):

    def __init__(self, scale=1, shift=0., scale_jitter=0., shift_jitter=0.):
        ImageAugmentation.__init__(self)
        self.scale = scale
        self.shift = shift
        self.scale_jitter = scale_jitter
        self.shift_jitter = shift_jitter

    def apply_image(self, image):
        scale = self.scale + self.scale_jitter * np.random.randn()
        shift = self.shift + self.shift_jitter * np.random.randn()
        return scale * image - shift


class NormalizeRange(ImageAugmentation):

    def __init__(self):
        ImageAugmentation.__init__(self)

    def apply_image(self, image):
        return (image - image.min()) / (image.max() - image.min())


class Normalize(ImageAugmentation):

    def __init__(self):
        ImageAugmentation.__init__(self)

    def apply_image(self, image):
        return (image - np.mean(image)) / np.std(image)


class LocalNormalize(ImageAugmentation):

    def __init__(self, sigma1, sigma2):
        ImageAugmentation.__init__(self)
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def apply_image(self, image):
        mean = gaussian_filter(image, self.sigma1)
        image = image - mean
        image = image / np.sqrt(gaussian_filter(image ** 2, self.sigma2))
        return image


class GaussianBlur(ImageAugmentation):

    def __init__(self, sigma, sigma_jitter):
        ImageAugmentation.__init__(self)
        self.sigma = sigma
        self.sigma_jitter = sigma_jitter

    def apply_image(self, image):
        sigma = np.max((self.sigma + np.random.randn() * self.sigma_jitter, 0.))
        if sigma > 0.:
            return gaussian_filter(image, sigma)
        else:
            return image


class Gamma(ImageAugmentation):

    def __init__(self, gamma=1, gamma_jitter=0.):
        ImageAugmentation.__init__(self)
        self.gamma = gamma
        self.gamma_jitter = gamma_jitter

    def apply_image(self, image):
        return image ** max(self.gamma + self.gamma_jitter * np.random.randn(), 0.)


class PoissonNoise(ImageAugmentation):

    def __init__(self, mean, mean_jitter=0.):
        ImageAugmentation.__init__(self)
        self.mean = mean
        self.mean_jitter = mean_jitter

    def apply_image(self, image):
        if image.min() < 0:
            raise RuntimeError()

        scale = np.max(((self.mean + np.random.randn() * self.mean_jitter), 1))
        mean = np.mean(image)
        image = scale * image / mean
        image = np.random.poisson(image).astype(np.float32)
        return image / scale * mean


class GaussianNoise(ImageAugmentation):

    def __init__(self, amount, amount_jitter=0.):
        ImageAugmentation.__init__(self)
        self.amount = amount
        self.amount_jitter = amount_jitter

    def apply_image(self, image):
        amount = self.amount + np.random.randn() * self.amount_jitter
        image = image + np.random.randn(*image.shape) * amount
        return image


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


class ScanNoise(ImageAugmentation):

    def __init__(self, scale, amount, fine_amount=None):
        ImageAugmentation.__init__(self)
        self.scale = scale
        self.amount = amount
        if fine_amount is None:
            self.fine_amount = self.amount / 2

    def apply_image(self, image):
        fine_scale = np.max(image.shape[:-1])

        n = ((self.amount * bandpass_noise(0, self.scale, image.shape[1]) +
              self.amount / 2 * bandpass_noise(0, fine_scale, image.shape[1]))).astype(int)

        def strided_indexing_roll(a, r):
            a_ext = np.concatenate((a, a[:, :-1]), axis=1)
            n = a.shape[1]
            return view_as_windows(a_ext, (1, n))[np.arange(len(r)), (n - r) % n, 0]

        for i in range(image.shape[-1]):
            image[..., i] = strided_indexing_roll(image[..., i], n)

        return image


class Dirt(CommonAugmentation):

    def __init__(self, scale, fraction, noise_scale=None):
        CommonAugmentation.__init__(self)
        self.scale = scale
        self.fraction = fraction
        if noise_scale is None:
            self.noise_scale = scale * 10
        else:
            self.noise_scale = noise_scale

    def _get_mask(self, shape):
        mask = bandpass_noise_2d(0, self.scale, shape)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask[mask > self.fraction] = self.fraction
        # mask = mask > self.fraction
        return (mask - mask.min()) / (mask.max() - mask.min())

    def _get_noise(self, shape):
        noise = bandpass_noise_2d(0, self.noise_scale, shape)
        return (noise - noise.min()) / (noise.max() - noise.min())

    def apply_image_and_label(self, image, label=None):
        mask = self._get_mask(image.shape[:-1])
        noise = self._get_noise(image.shape[:-1])
        std = image.std()

        image = image * mask[..., None] / std + noise[..., None] * (1 - mask[..., None]) / noise.std() * 2

        image *= std

        # label[..., 1:] *= mask[..., None] ** 5
        # label[..., 0] = 1 - np.sum(label[..., 1:], axis=2)

        return image, label
