import numpy as np
from ase import units
from scipy.special import erfc


def lobato_potential(parameters):
    a = [np.pi ** 2 * parameters[key_a] / parameters[key_b] ** (3 / 2.) for key_a, key_b in
         zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
    b = [2 * np.pi / np.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    v = lambda r: (a[0] * (2. / (b[0] * r) + 1) * np.exp(-b[0] * r) +
                   a[1] * (2. / (b[1] * r) + 1) * np.exp(-b[1] * r) +
                   a[2] * (2. / (b[2] * r) + 1) * np.exp(-b[2] * r) +
                   a[3] * (2. / (b[3] * r) + 1) * np.exp(-b[3] * r) +
                   a[4] * (2. / (b[4] * r) + 1) * np.exp(-b[4] * r))

    dvdr = lambda r: - (a[0] * (2 / (b[0] * r ** 2) + 2 / r + b[0]) * np.exp(-b[0] * r) +
                        a[1] * (2 / (b[1] * r ** 2) + 2 / r + b[1]) * np.exp(-b[1] * r) +
                        a[2] * (2 / (b[2] * r ** 2) + 2 / r + b[2]) * np.exp(-b[2] * r) +
                        a[3] * (2 / (b[3] * r ** 2) + 2 / r + b[3]) * np.exp(-b[3] * r) +
                        a[4] * (2 / (b[4] * r ** 2) + 2 / r + b[4]) * np.exp(-b[4] * r))

    return v, dvdr


def lobato_scattering_factor(parameters):
    a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4', 'a5')]
    b = [parameters[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    f = lambda g: (a[0] * (2 + b[0] * g ** 2) / (1 + b[0] * g ** 2) ** 2 +
                   a[1] * (2 + b[1] * g ** 2) / (1 + b[1] * g ** 2) ** 2 +
                   a[2] * (2 + b[2] * g ** 2) / (1 + b[2] * g ** 2) ** 2 +
                   a[3] * (2 + b[3] * g ** 2) / (1 + b[3] * g ** 2) ** 2 +
                   a[4] * (2 + b[4] * g ** 2) / (1 + b[4] * g ** 2) ** 2)

    dfdg = lambda g: - ((2 * a[0] * b[0] * g * (3 + b[0] * g ** 2)) / (1 + b[0] * g ** 2) ** 3 +
                        (2 * a[1] * b[1] * g * (3 + b[1] * g ** 2)) / (1 + b[1] * g ** 2) ** 3 +
                        (2 * a[2] * b[2] * g * (3 + b[2] * g ** 2)) / (1 + b[2] * g ** 2) ** 3 +
                        (2 * a[3] * b[3] * g * (3 + b[3] * g ** 2)) / (1 + b[3] * g ** 2) ** 3 +
                        (2 * a[4] * b[4] * g * (3 + b[4] * g ** 2)) / (1 + b[4] * g ** 2) ** 3)

    return f, dfdg


def lobato_density(parameters):
    a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4', 'a5')]
    b = [parameters[key] for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    rho = lambda r: 2 * np.pi ** 4 * units.Bohr * (a[0] / b[0] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[0])) +
                                                   a[1] / b[1] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[1])) +
                                                   a[2] / b[2] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[2])) +
                                                   a[3] / b[3] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[3])) +
                                                   a[4] / b[4] ** (5 / 2.) * np.exp(-2 * np.pi * r / np.sqrt(b[4])))

    return rho


def kirkland_potential(parameters):
    a = [np.pi * parameters[key] for key in ('a1', 'a2', 'a3')]
    b = [2 * np.pi * np.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3')]
    c = [np.pi ** (3 / 2.) * parameters[key_c] / parameters[key_d] ** (3 / 2.) for key_c, key_d in
         zip(('c1', 'c2', 'c3'), ('d1', 'd2', 'd3'))]
    d = [np.pi ** 2 / parameters[key] for key in ('d1', 'd2', 'd3')]

    v = lambda r: (a[0] * np.exp(-b[0] * r) / r + c[0] * np.exp(-d[0] * r ** 2) +
                   a[1] * np.exp(-b[1] * r) / r + c[1] * np.exp(-d[1] * r ** 2) +
                   a[2] * np.exp(-b[2] * r) / r + c[2] * np.exp(-d[2] * r ** 2))

    dvdr = lambda r: (- a[0] * (1 / r + b[0]) * np.exp(-b[0] * r) / r - 2 * c[0] * d[0] * r * np.exp(-d[0] * r ** 2)
                      - a[1] * (1 / r + b[1]) * np.exp(-b[1] * r) / r - 2 * c[1] * d[1] * r * np.exp(-d[1] * r ** 2)
                      - a[2] * (1 / r + b[2]) * np.exp(-b[2] * r) / r - 2 * c[2] * d[2] * r * np.exp(-d[2] * r ** 2)
                      )

    return v, dvdr


def kirkland_scattering_factor(parameters):
    a = [parameters[key] for key in ('a1', 'a2', 'a3')]
    b = [parameters[key] for key in ('b1', 'b2', 'b3')]
    c = [parameters[key] for key in ('c1', 'c2', 'c3')]
    d = [parameters[key] for key in ('d1', 'd2', 'd3')]

    f = lambda g: (a[0] / (b[0] + g ** 2) + c[0] * np.exp(-d[0] * g ** 2) +
                   a[1] / (b[1] + g ** 2) + c[1] * np.exp(-d[1] * g ** 2) +
                   a[2] / (b[2] + g ** 2) + c[2] * np.exp(-d[2] * g ** 2))

    dfdg = lambda g: (- 2 * a[0] * g / (b[0] + g ** 2) ** 2 - 2 * c[0] * d[0] * g * np.exp(-d[0] * g ** 2)
                      - 2 * a[1] * g / (b[1] + g ** 2) ** 2 - 2 * c[1] * d[1] * g * np.exp(-d[1] * g ** 2)
                      - 2 * a[2] * g / (b[2] + g ** 2) ** 2 - 2 * c[2] * d[2] * g * np.exp(-d[2] * g ** 2))

    return f, dfdg


def gaussian_sum_potential(parameters):
    a = [np.pi ** (3 / 2.) * parameters[key_a] / (parameters[key_b] / 4) ** (3 / 2.) for key_a, key_b in
         zip(('a1', 'a2', 'a3', 'a4'), ('b1', 'b2', 'b3', 'b4'))]

    b = [np.pi ** 2 / (parameters[key] / 4) for key in ('b1', 'b2', 'b3', 'b4')]

    v = lambda r: (a[0] * np.exp(-b[0] * r ** 2) +
                   a[1] * np.exp(-b[1] * r ** 2) +
                   a[2] * np.exp(-b[2] * r ** 2) +
                   a[3] * np.exp(-b[3] * r ** 2))

    dvdr = lambda r: (- 2 * a[0] * b[0] * r * np.exp(-b[0] * r ** 2)
                      - 2 * a[1] * b[1] * r * np.exp(-b[1] * r ** 2)
                      - 2 * a[2] * b[2] * r * np.exp(-b[2] * r ** 2)
                      - 2 * a[3] * b[3] * r * np.exp(-b[3] * r ** 2))

    return v, dvdr


def gaussian_sum_scattering_factor(parameters):
    a = [parameters[key] for key in ('a1', 'a2', 'a3', 'a4')]
    b = [parameters[key] / 4 for key in ('b1', 'b2', 'b3', 'b4')]

    f = lambda g: (a[0] * np.exp(-b[0] * g ** 2) +
                   a[1] * np.exp(-b[1] * g ** 2) +
                   a[2] * np.exp(-b[2] * g ** 2) +
                   a[3] * np.exp(-b[3] * g ** 2))

    dfdg = lambda g: (- 2 * a[0] * b[0] * np.exp(-b[0] * g ** 2)
                      - 2 * a[1] * b[1] * np.exp(-b[1] * g ** 2)
                      - 2 * a[2] * b[2] * np.exp(-b[2] * g ** 2)
                      - 2 * a[3] * b[3] * np.exp(-b[3] * g ** 2))

    return f, dfdg


def weickenmeier_potential(parameters):
    a = 3 * [4 * np.pi * 0.02395 * parameters['Z'] / (3 * (1 + parameters['V']))]
    a = a + 3 * [parameters['V'] * a[0]]
    b = [np.pi / np.sqrt(parameters[key] / 4) for key in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')]

    v = lambda r: (a[0] * erfc(b[0] * r) / r +
                   a[1] * erfc(b[1] * r) / r +
                   a[2] * erfc(b[2] * r) / r +
                   a[3] * erfc(b[3] * r) / r +
                   a[4] * erfc(b[4] * r) / r +
                   a[5] * erfc(b[5] * r) / r)

    dvdr = lambda r: (- a[0] * (erfc(b[0] * r) / r ** 2 + 2 * b[0] / np.sqrt(np.pi) * np.exp(-b[0] * r ** 2) / r)
                      - a[1] * (erfc(b[1] * r) / r ** 2 + 2 * b[1] / np.sqrt(np.pi) * np.exp(-b[1] * r ** 2) / r)
                      - a[2] * (erfc(b[2] * r) / r ** 2 + 2 * b[2] / np.sqrt(np.pi) * np.exp(-b[2] * r ** 2) / r)
                      - a[3] * (erfc(b[3] * r) / r ** 2 + 2 * b[3] / np.sqrt(np.pi) * np.exp(-b[3] * r ** 2) / r)
                      - a[4] * (erfc(b[4] * r) / r ** 2 + 2 * b[4] / np.sqrt(np.pi) * np.exp(-b[4] * r ** 2) / r)
                      - a[5] * (erfc(b[5] * r) / r ** 2 + 2 * b[5] / np.sqrt(np.pi) * np.exp(-b[5] * r ** 2) / r))

    return v, dvdr


def weickenmeier_scattering_factor(parameters):
    a = 3 * [4 * 0.02395 * parameters['Z'] / (3 * (1 + parameters['V']))]
    a = a + 3 * [parameters['V'] * a[0]]
    b = [parameters[key] / 4 for key in ('b1', 'b2', 'b3', 'b4', 'b5', 'b6')]

    f = lambda g: (a[0] * (1 - np.exp(-b[0] * g ** 2)) / g ** 2 +
                   a[1] * (1 - np.exp(-b[1] * g ** 2)) / g ** 2 +
                   a[2] * (1 - np.exp(-b[2] * g ** 2)) / g ** 2 +
                   a[3] * (1 - np.exp(-b[3] * g ** 2)) / g ** 2 +
                   a[4] * (1 - np.exp(-b[4] * g ** 2)) / g ** 2 +
                   a[5] * (1 - np.exp(-b[5] * g ** 2)) / g ** 2)

    dfdg = lambda g: 1

    return f, dfdg


potential_dict = {'lobato': {'potential': lobato_potential,
                             'scattering_factor': lobato_scattering_factor,
                             'density': lobato_density,
                             'default_parameters': 'data/lobato.txt'},
                  'kirkland': {'potential': kirkland_potential,
                               'scattering_factor': kirkland_scattering_factor,
                               'density': None,
                               'default_parameters': 'data/kirkland.txt'},
                  'peng': {'potential': gaussian_sum_potential,
                           'scattering_factor': gaussian_sum_scattering_factor,
                           'density': None,
                           'default_parameters': 'data/peng.txt'},
                  'weickenmeier': {'potential': weickenmeier_potential,
                                   'scattering_factor': weickenmeier_scattering_factor,
                                   'density': None,
                                   'default_parameters': 'data/weickenmeier.txt'},
                  'gpaw': {'potential': kirkland_potential,
                           'scattering_factor': kirkland_scattering_factor,
                           'density': None,
                           'default_parameters': 'data/gpaw.txt'}
                  }
