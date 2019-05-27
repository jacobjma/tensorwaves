
#
#
# def zernike_polynomial(rho, phi, n, m):
#     assert n >= m
#     assert (n + m) % 2 == 0
#
#     if m >= 0:
#         even = True
#     else:
#         even = False
#
#     def factorial(n):
#         return np.prod(range(1, n + 1)).astype(int)
#
#     def normalization(n, m, k):
#         return (-1) ** k * factorial(n - k) // (
#                 factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
#
#     m = abs(m)
#
#     R = np.zeros_like(rho)
#     for k in range(0, (n - m) // 2 + 1):
#         if (n - 2 * k) > 1:
#             R += normalization(n, m, k) * rho ** (n - 2 * k)
#
#     if even:
#         Z = R * np.cos(m * phi)
#     else:
#         Z = R * np.sin(m * phi)
#
#     return Z
#
#
# class ZernikeExpansion(object):
#
#     def __init__(self, coefficients, basis, indices):
#         self._coefficients = coefficients
#         self._basis = basis
#         self._indices = indices
#
#     def to_parametrization(self):
#         pass
#
#     def sum(self):
#         return tf.reduce_sum(self._basis * self._coefficients[:, None], axis=0)
#
#
# class ZernikeAberrations(Parametrization):
#
#     def __init__(self, aperture_radius, max_order=6, parameters=None):
#
#         self._aperture_radius = aperture_radius
#
#         if parameters is None:
#             parameters = {}
#
#         aliases = {}
#         symbols = []
#
#         symmetric = False
#
#         for n in range(1, max_order + 1):
#             for m in range(-n, n + 1):
#                 if (not symmetric) | (m == 0):
#                     if (n - m) % 2 == 0:
#                         symbols.append((n, m))
#
#         Parametrization.__init__(self, symbols, aliases, parameters)
#
#     @property
#     def aperture_radius(self):
#         return self._aperture_radius
#
#     def expansion(self, k, phi):
#         k /= self._aperture_radius
#
#         indices = []
#         expansion = []
#         coefficients = []
#         for (n, m), value in self.parameters.items():
#             indices.append((n, m))
#             expansion.append(zernike_polynomial(k, phi, n, m))
#             coefficients.append(value)
#
#         expansion = tf.convert_to_tensor(expansion)
#         coefficients = tf.convert_to_tensor(coefficients)
#
#         return indices, expansion, coefficients
#
#     def to_polar(self):
#         parameters = zernike2polar(self.parameters, self.aperture_radius)
#         return PolarAberrations(parameters=parameters)
#
#     def __call__(self, k, phi):
#         k /= self._aperture_radius
#
#         Z = tf.zeros(k.shape)
#         for (n, m), value in self.parameters.items():
#             Z += value * zernike_polynomial(k, phi, n, m)
#         return Z
#
#
# def polar_aberration_order(symbol):
#     for letter in list(symbol):
#         try:
#             return int(letter)
#         except:
#             pass
#
#
# def polar2zernike(polar, aperture_radius):
#     polar = defaultdict(lambda: 0, polar)
#
#     for symbol, value in polar.items():
#         if symbol[0] == 'C':
#             polar[symbol] *= aperture_radius ** (polar_aberration_order(symbol) + 1)
#
#     zernike = {}
#     zernike[(1, 1)] = 2 * polar['C21'] / 9. * np.cos(polar['phi21']) + polar['C41'] / 10. * np.cos(polar['phi41'])
#     zernike[(1, -1)] = 2 * polar['C21'] / 9. * np.sin(polar['phi21']) + polar['C41'] / 10. * np.sin(polar['phi41'])
#
#     zernike[(2, 0)] = polar['C10'] / 4. + polar['C30'] / 8. + 3 / 40. * polar['C50']
#     zernike[(2, 2)] = polar['C12'] / 2. * np.cos(2 * polar['phi12']) + \
#                       3 * polar['C32'] / 16. * np.cos(2 * polar['phi32']) + \
#                       (1 / 6. - 1 / 15.) * polar['C52'] * np.cos(2 * polar['phi52'])
#     zernike[(2, -2)] = polar['C12'] / 2. * np.sin(2 * polar['phi12']) + \
#                        3 * polar['C32'] / 16. * np.sin(2 * polar['phi32']) + \
#                        (1 / 6. - 1 / 15.) * polar['C52'] * np.sin(2 * polar['phi52'])
#
#     zernike[(3, 1)] = polar['C21'] / 9. * np.cos(polar['phi21']) + 4 * polar['C41'] / 50. * np.cos(polar['phi41'])
#     zernike[(3, -1)] = polar['C21'] / 9. * np.sin(polar['phi21']) + 4 * polar['C41'] / 50. * np.sin(polar['phi41'])
#     zernike[(3, 3)] = polar['C23'] / 3. * np.cos(3 * polar['phi23']) + 4 * polar['C43'] / 25. * np.cos(
#         3 * polar['phi43'])
#     zernike[(3, -3)] = polar['C23'] / 3. * np.sin(3 * polar['phi23']) + 4 * polar['C43'] / 25. * np.sin(
#         3 * polar['phi43'])
#
#     zernike[(4, 0)] = polar['C30'] / 24. + polar['C50'] / 24.
#     zernike[(4, 2)] = polar['C32'] / 16. * np.cos(2 * polar['phi32']) + polar['C52'] / 18. * np.cos(2 * polar['phi52'])
#     zernike[(4, -2)] = polar['C32'] / 16. * np.sin(2 * polar['phi32']) + polar['C52'] / 18. * np.sin(2 * polar['phi52'])
#     zernike[(4, 4)] = polar['C34'] / 4. * np.cos(4 * polar['phi34']) + 5 * polar['C54'] / 36. * np.cos(
#         4 * polar['phi54'])
#     zernike[(4, -4)] = polar['C34'] / 4. * np.sin(4 * polar['phi34']) + 5 * polar['C54'] / 36. * np.sin(
#         4 * polar['phi54'])
#
#     zernike[(5, 1)] = polar['C41'] / 50. * np.cos(polar['phi41'])
#     zernike[(5, -1)] = polar['C41'] / 50. * np.sin(polar['phi41'])
#     zernike[(5, 3)] = polar['C43'] / 25. * np.cos(3 * polar['phi43'])
#     zernike[(5, -3)] = polar['C43'] / 25. * np.sin(3 * polar['phi43'])
#     zernike[(5, 5)] = polar['C45'] / 5. * np.cos(5 * polar['phi45'])
#     zernike[(5, -5)] = polar['C45'] / 5. * np.sin(5 * polar['phi45'])
#
#     zernike[(6, 0)] = polar['C50'] / 120.
#     zernike[(6, 2)] = polar['C52'] / 90. * np.cos(2 * polar['phi52'])
#     zernike[(6, -2)] = polar['C52'] / 90. * np.sin(2 * polar['phi52'])
#     zernike[(6, 4)] = polar['C54'] / 36. * np.cos(4 * polar['phi54'])
#     zernike[(6, -4)] = polar['C54'] / 36. * np.sin(4 * polar['phi54'])
#     zernike[(6, 6)] = polar['C56'] / 6. * np.cos(6 * polar['phi56'])
#     zernike[(6, -6)] = polar['C56'] / 6. * np.sin(6 * polar['phi56'])
#
#     return {key: value for key, value in zernike.items() if value != 0.}
#
#
# def zernike2polar(zernike, aperture_radius):
#     zernike = defaultdict(lambda: 0., zernike)
#
#     polar = {}
#     polar['C50'] = 120 * zernike[(6, 0)]
#     polar['C52'] = np.sqrt(zernike[(6, -2)] ** 2 + zernike[(6, 2)] ** 2) * 90
#     polar['phi52'] = np.arctan2(zernike[(6, -2)], zernike[(6, 2)]) / 2.
#     polar['C54'] = np.sqrt(zernike[(6, -4)] ** 2 + zernike[(6, 4)] ** 2) * 36
#     polar['phi54'] = np.arctan2(zernike[(6, -4)], zernike[(6, 4)]) / 4.
#     polar['C56'] = np.sqrt(zernike[(6, -6)] ** 2 + zernike[(6, 6)] ** 2) * 6
#     polar['phi56'] = np.arctan2(zernike[(6, -6)], zernike[(6, 6)]) / 6.
#
#     polar['C41'] = np.sqrt(zernike[(5, -1)] ** 2 + zernike[(5, 1)] ** 2) * 50
#     polar['phi41'] = np.arctan2(zernike[(5, -1)], zernike[(5, 1)])
#     polar['C43'] = np.sqrt(zernike[(5, -3)] ** 2 + zernike[(5, 3)] ** 2) * 25
#     polar['phi43'] = np.arctan2(zernike[(5, -3)], zernike[(5, 3)]) / 3.
#     polar['C45'] = np.sqrt(zernike[(5, -5)] ** 2 + zernike[(5, 5)] ** 2) * 5
#     polar['phi45'] = np.arctan2(zernike[(5, -5)], zernike[(5, 5)]) / 5.
#
#     polar['C30'] = 24 * zernike[(4, 0)] - polar['C50']
#     polar['C32'] = np.sqrt((zernike[(4, -2)] - polar['C52'] / 18. * np.sin(2 * polar['phi52'])) ** 2 +
#                            (zernike[(4, 2)] - polar['C52'] / 18. * np.cos(2 * polar['phi52'])) ** 2) * 16
#     polar['phi32'] = np.arctan2(zernike[(4, -2)] - polar['C52'] / 18. * np.sin(2 * polar['phi52']),
#                                 zernike[(4, 2)] - polar['C52'] / 18. * np.cos(2 * polar['phi52'])) / 2.
#     polar['C34'] = np.sqrt((zernike[(4, -4)] - 5 * polar['C54'] / 36. * np.sin(4 * polar['phi54'])) ** 2 +
#                            (zernike[(4, 4)] - 5 * polar['C54'] / 36. * np.cos(4 * polar['phi54'])) ** 2) * 4
#     polar['phi34'] = np.arctan2(zernike[(4, -4)] - 5 * polar['C54'] / 36. * np.sin(4 * polar['phi54']),
#                                 zernike[(4, 4)] - 5 * polar['C54'] / 36. * np.cos(4 * polar['phi54'])) / 4.
#
#     polar['C21'] = np.sqrt((zernike[(3, -1)] - 4 * polar['C41'] / 50. * np.sin(polar['phi41'])) ** 2 +
#                            (zernike[(3, 1)] - 4 * polar['C41'] / 50. * np.cos(polar['phi41'])) ** 2) * 9
#     polar['phi21'] = np.arctan2(zernike[(3, -1)] - 4 * polar['C41'] / 50. * np.sin(polar['phi41']),
#                                 zernike[(3, 1)] - 4 * polar['C41'] / 50. * np.cos(polar['phi41']))
#     polar['C23'] = np.sqrt((zernike[(3, -3)] - 4 * polar['C43'] / 25. * np.sin(3 * polar['phi43'])) ** 2 +
#                            (zernike[(3, 3)] - 4 * polar['C43'] / 25. * np.cos(3 * polar['phi43'])) ** 2) * 3
#     polar['phi23'] = np.arctan2(zernike[(3, -3)] - 4 * polar['C43'] / 25. * np.sin(3 * polar['phi43']),
#                                 zernike[(3, 3)] - 4 * polar['C43'] / 25. * np.cos(3 * polar['phi43'])) / 3.
#
#     polar['C10'] = 4 * zernike[(2, 0)] - polar['C30'] / 2. - 3 / 10. * polar['C50']
#     polar['C12'] = np.sqrt((zernike[(2, -2)]
#                             - 3 * polar['C32'] / 16. * np.sin(2 * polar['phi32'])
#                             - (1 / 6. - 1 / 15.) * polar['C52'] * np.sin(2 * polar['phi52'])) ** 2 +
#                            (zernike[(2, 2)]
#                             - 3 * polar['C32'] / 16. * np.cos(2 * polar['phi32'])
#                             - (1 / 6. - 1 / 15.) * polar['C52'] * np.cos(2 * polar['phi52'])) ** 2) * 2
#     polar['phi12'] = np.arctan2(zernike[(2, -2)]
#                                 - 3 * polar['C32'] / 16. * np.sin(2 * polar['phi32'])
#                                 - (1 / 6. - 1 / 15.) * polar['C52'] * np.sin(2 * polar['phi52']),
#                                 zernike[(2, 2)]
#                                 - 3 * polar['C32'] / 16. * np.cos(2 * polar['phi32'])
#                                 - (1 / 6. - 1 / 15.) * polar['C52'] * np.cos(2 * polar['phi52'])) / 2.
#
#     for symbol, value in polar.items():
#         if symbol[0] == 'C':
#             polar[symbol] /= aperture_radius ** (polar_aberration_order(symbol) + 1)
#
#     return {key: value for key, value in polar.items() if value != 0.}
#
#
