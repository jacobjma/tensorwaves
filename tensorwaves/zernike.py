from collections import defaultdict

import numpy as np


def zernike_polynomial(rho, phi, n, m):
    assert n >= m
    assert (n + m) % 2 == 0

    if m >= 0:
        even = True
    else:
        even = False

    def factorial(n):
        return np.prod(range(1, n + 1)).astype(int)

    def normalization(n, m, k):
        return (-1) ** k * factorial(n - k) // (
                factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))

    m = abs(m)

    R = np.zeros_like(rho)
    for k in range(0, (n - m) // 2 + 1):
        if (n - 2 * k) > 0:
            R += normalization(n, m, k) * rho ** (n - 2 * k)

    if even:
        Z = R * np.cos(m * phi)
    else:
        Z = R * np.sin(m * phi)

    return Z


class ZernikeExpansion(object):

    def __init__(self, x, y, coefficients, max_order=None):
        self._x = x
        self._y = y

        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(x, y)

        self._basis = np.zeros((len(coefficients), len(x)), dtype=np.float32)
        self._coefficients = np.zeros(len(coefficients), dtype=np.float32)
        self._indices = []

        for i, ((n, m), value) in enumerate(coefficients.items()):
            self._basis[i] = zernike_polynomial(r, phi, n, m)
            self._coefficients[i] = value
            self._indices.append((n, m))

    def sum(self):
        return np.sum(self._basis * self._coefficients[:, None], axis=0)
