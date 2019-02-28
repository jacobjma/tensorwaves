import numpy as np
import scipy.ndimage
from skimage.util.shape import view_as_windows


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

        if zoom < 1.:
            new_zoomed = np.zeros(shape)
            new_zoomed[:zoomed.shape[0], :zoomed.shape[1]] = zoomed  # [:shape[0], :shape[1]]
            return new_zoomed
        else:
            return zoomed[:shape[0], :shape[1]]

    def apply_image_and_label(self, image, label):
        zoom = self.zoom + self.zoom_jitter * np.random.randn()
        return self._apply(image, zoom), self._apply(label, zoom)


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
        scale = np.max(((self.mean + np.random.randn() * self.mean_jitter), 1))
        image = scale * (image - image.min()) / (image.max() - image.min())
        image = np.random.poisson(image).astype(np.float32)
        return image / scale


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
        return (mask - mask.min()) / (mask.max() - mask.min())

    def _get_noise(self, shape):
        noise = bandpass_noise_2d(0, self.noise_scale, shape)
        return (noise - noise.min()) / (noise.max() - noise.min())

    def apply_image_and_label(self, image, label=None):
        mask = self._get_mask(image.shape[:-1])
        noise = self._get_noise(image.shape[:-1])
        image = image * mask[..., None] / image.std() + noise[..., None] * (1 - mask[..., None]) / noise.std() * 3

        label[..., :-1] *= mask[..., None]
        label[..., -1] = 1 - np.sum(label[..., :-1], axis=2)

        return image, label
