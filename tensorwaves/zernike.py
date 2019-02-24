from collections import defaultdict

import numpy as np


def factorial(n):
    return np.prod(range(1, n + 1)).astype(int)


def binomial(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def zernike_coefficient(n, m, k):
    return (-1) ** k * factorial(n - k) // (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))


def zernike_polynomial(rho, phi, n, m):
    assert n >= m
    assert (n + m) % 2 == 0

    if m >= 0:
        even = True
    else:
        even = False

    m = abs(m)

    R = np.zeros_like(rho)
    for k in range(0, (n - m) // 2 + 1):
        if (n - 2 * k) > 0:
            R += zernike_coefficient(n, m, k) * rho ** (n - 2 * k)

    if even:
        Z = R * np.cos(m * phi)
    else:
        Z = R * np.sin(m * phi)

    return Z


def polar_aberration_order(symbol):
    for letter in list(symbol):
        try:
            return int(letter)
        except:
            pass


def polar2zernike(polar, aperture_radius):
    polar = defaultdict(lambda: 0, polar)

    for symbol, value in polar.items():
        if symbol[0] == 'C':
            polar[symbol] *= aperture_radius ** (polar_aberration_order(symbol) + 1)

    zernike = {}
    zernike[(1, 1)] = 2 * (polar['C21'] / 9 * np.cos(polar['phi21']) +
                           polar['C41'] / 50 * np.cos(polar['phi41']) * 12 / 3)
    zernike[(1, -1)] = 2 * (polar['C21'] / 9 * np.sin(polar['phi21']) +
                            polar['C41'] / 50 * np.sin(polar['phi41']) * 12 / 3)

    zernike[(1, 1)] -= 3 * polar['C41'] / 50 * np.cos(polar['phi41'])
    zernike[(1, -1)] -= 3 * polar['C41'] / 50 * np.sin(polar['phi41'])

    zernike[(2, 0)] = polar['C10'] / 4 + polar['C30'] / 8 + 3 / 40 * polar['C50']
    zernike[(2, 2)] = polar['C12'] / 2 * np.cos(2 * polar['phi12'])
    zernike[(2, -2)] = polar['C12'] / 2 * np.sin(2 * polar['phi12'])

    zernike[(2, 2)] += 3 * polar['C32'] / 16 * np.cos(2 * polar['phi32']) + polar['C52'] / 6 * np.cos(
        2 * polar['phi52'])

    zernike[(2, -2)] += (polar['C32'] / 16 * np.sin(2 * polar['phi32'])
                         + polar['C52'] / 90 * np.sin(2 * polar['phi52']) * 5) * 3

    zernike[(2, 2)] -= polar['C52'] / 90 * np.cos(2 * polar['phi52']) * 6
    zernike[(2, -2)] -= polar['C52'] / 90 * np.sin(2 * polar['phi52']) * 6

    zernike[(3, 1)] = polar['C21'] / 9 * np.cos(polar['phi21'])
    zernike[(3, -1)] = polar['C21'] / 9 * np.sin(polar['phi21'])
    zernike[(3, 1)] += polar['C41'] / 50 * np.cos(polar['phi41']) * 12 / 3
    zernike[(3, -1)] += polar['C41'] / 50 * np.sin(polar['phi41']) * 12 / 3

    zernike[(3, 3)] = polar['C23'] / 3 * np.cos(3 * polar['phi23'])
    zernike[(3, -3)] = polar['C23'] / 3 * np.sin(3 * polar['phi23'])
    zernike[(3, 3)] += polar['C43'] / 25 * np.cos(3 * polar['phi43']) * 4
    zernike[(3, -3)] += polar['C43'] / 25 * np.sin(3 * polar['phi43']) * 4

    zernike[(4, 0)] = polar['C30'] / 24 + polar['C50'] / 24
    zernike[(4, 2)] = polar['C32'] / 16 * np.cos(2 * polar['phi32'])
    zernike[(4, -2)] = polar['C32'] / 16 * np.sin(2 * polar['phi32'])

    zernike[(4, 2)] += polar['C52'] / 90 * np.cos(2 * polar['phi52']) * 5
    zernike[(4, -2)] += polar['C52'] / 90 * np.sin(2 * polar['phi52']) * 5

    zernike[(4, 4)] = polar['C34'] / 4 * np.cos(4 * polar['phi34'])
    zernike[(4, -4)] = polar['C34'] / 4 * np.sin(4 * polar['phi34'])
    zernike[(4, 4)] += polar['C54'] / 36 * np.cos(4 * polar['phi54']) * 5
    zernike[(4, -4)] += polar['C54'] / 36 * np.sin(4 * polar['phi54']) * 5

    zernike[(5, 1)] = polar['C41'] / 50 * np.cos(polar['phi41'])
    zernike[(5, -1)] = polar['C41'] / 50 * np.sin(polar['phi41'])
    zernike[(5, 3)] = polar['C43'] / 25 * np.cos(3 * polar['phi43'])
    zernike[(5, -3)] = polar['C43'] / 25 * np.sin(3 * polar['phi43'])
    zernike[(5, 5)] = polar['C45'] / 5 * np.cos(5 * polar['phi45'])
    zernike[(5, -5)] = polar['C45'] / 5 * np.sin(5 * polar['phi45'])

    zernike[(6, 0)] = polar['C50'] / 120
    zernike[(6, 2)] = polar['C52'] / 90 * np.cos(2 * polar['phi52'])
    zernike[(6, -2)] = polar['C52'] / 90 * np.sin(2 * polar['phi52'])
    zernike[(6, 4)] = polar['C54'] / 36 * np.cos(4 * polar['phi54'])
    zernike[(6, -4)] = polar['C54'] / 36 * np.sin(4 * polar['phi54'])
    zernike[(6, 6)] = polar['C56'] / 6 * np.cos(6 * polar['phi56'])
    zernike[(6, -6)] = polar['C56'] / 6 * np.sin(6 * polar['phi56'])

    return {key: value for key, value in zernike.items() if value != 0.}


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
