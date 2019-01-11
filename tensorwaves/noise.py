import numpy as np
from tensorwaves.utils import fftfreq2d


def spectral_noise(gpts, sampling, func):
    kx, ky = fftfreq2d(gpts, sampling)
    k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)

    abs_F = func(k)
    abs_F[abs_F < 0] = 0
    abs_F = np.sqrt(abs_F)

    v = np.random.rand(abs_F.shape[0], abs_F.shape[1])

    F = abs_F * np.exp(-1.j * 2 * np.pi * v)

    return (np.fft.ifft2(F).real + np.fft.ifft2(F).imag) / 2


def power_law_noise(gpts, sampling, power):
    def fit_func(k):
        ret_vals = np.zeros_like(k)
        ret_vals[k != 0] = k[k != 0] ** power
        return ret_vals

    noise = spectral_noise(gpts, sampling, fit_func)

    #import matplotlib.pyplot as plt
    #plt.imshow(noise)
    #plt.show()

    return noise
