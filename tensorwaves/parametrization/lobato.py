from math import pi
import tensorflow as tf
from scipy.special import kn


def potential(parameters):
    a = [pi ** 2 * parameters[key_a] / parameters[key_b] ** (3 / 2.) for key_a, key_b in
         zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
    b = [2 * pi / tf.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    def func(r):
        return (a[0] * (2. / (b[0] * r) + 1) * tf.exp(-b[0] * r) +
                a[1] * (2. / (b[1] * r) + 1) * tf.exp(-b[1] * r) +
                a[2] * (2. / (b[2] * r) + 1) * tf.exp(-b[2] * r) +
                a[3] * (2. / (b[3] * r) + 1) * tf.exp(-b[3] * r) +
                a[4] * (2. / (b[4] * r) + 1) * tf.exp(-b[4] * r))

    return func


def soft_potential(parameters, r_cut):
    a = [pi ** 2 * parameters[key_a] / parameters[key_b] ** (3 / 2.) for key_a, key_b in
         zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
    b = [2 * pi / tf.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    dvdr_cut = - (a[0] * (2 / (b[0] * r_cut ** 2) + 2 / r_cut + b[0]) * tf.exp(-b[0] * r_cut) +
                  a[1] * (2 / (b[1] * r_cut ** 2) + 2 / r_cut + b[1]) * tf.exp(-b[1] * r_cut) +
                  a[2] * (2 / (b[2] * r_cut ** 2) + 2 / r_cut + b[2]) * tf.exp(-b[2] * r_cut) +
                  a[3] * (2 / (b[3] * r_cut ** 2) + 2 / r_cut + b[3]) * tf.exp(-b[3] * r_cut) +
                  a[4] * (2 / (b[4] * r_cut ** 2) + 2 / r_cut + b[4]) * tf.exp(-b[4] * r_cut))

    v_cut = (a[0] * (2. / (b[0] * r_cut) + 1) * tf.exp(-b[0] * r_cut) +
             a[1] * (2. / (b[1] * r_cut) + 1) * tf.exp(-b[1] * r_cut) +
             a[2] * (2. / (b[2] * r_cut) + 1) * tf.exp(-b[2] * r_cut) +
             a[3] * (2. / (b[3] * r_cut) + 1) * tf.exp(-b[3] * r_cut) +
             a[4] * (2. / (b[4] * r_cut) + 1) * tf.exp(-b[4] * r_cut))

    def func(r):
        v = (a[0] * (2. / (b[0] * r) + 1) * tf.exp(-b[0] * r) +
             a[1] * (2. / (b[1] * r) + 1) * tf.exp(-b[1] * r) +
             a[2] * (2. / (b[2] * r) + 1) * tf.exp(-b[2] * r) +
             a[3] * (2. / (b[3] * r) + 1) * tf.exp(-b[3] * r) +
             a[4] * (2. / (b[4] * r) + 1) * tf.exp(-b[4] * r))

        return v - v_cut - (r - r_cut) * dvdr_cut

    return func


def projected_potential(parameters):
    a = [2. * pi ** 2 * parameters[key_a] / parameters[key_b] ** (3 / 2.) for key_a, key_b in
         zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
    b = [2 * pi / tf.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]

    def func(R):
        return (a[0] * (2. / b[0] * kn(0, b[0] * R) + R * kn(1, b[0] * R)) +
                a[1] * (2. / b[1] * kn(0, b[1] * R) + R * kn(1, b[1] * R)) +
                a[2] * (2. / b[2] * kn(0, b[2] * R) + R * kn(1, b[2] * R)) +
                a[3] * (2. / b[3] * kn(0, b[3] * R) + R * kn(1, b[3] * R)) +
                a[4] * (2. / b[4] * kn(0, b[4] * R) + R * kn(1, b[4] * R)))

    return func

# def soft_potential(parameters, rcut):
#     a = [pi ** 2 * parameters[key_a] / parameters[key_b] ** (3 / 2.) for key_a, key_b in
#          zip(('a1', 'a2', 'a3', 'a4', 'a5'), ('b1', 'b2', 'b3', 'b4', 'b5'))]
#     b = [2 * pi / tf.sqrt(parameters[key]) for key in ('b1', 'b2', 'b3', 'b4', 'b5')]
#
#     def v(R2, z2):
#         r2 = R2 + z2
#         r = tf.sqrt(r2)
#
#         v = (a[0] * (2. / (b[0] * r) + 1) * tf.exp(-b[0] * r) +
#              a[1] * (2. / (b[1] * r) + 1) * tf.exp(-b[1] * r) +
#              a[2] * (2. / (b[2] * r) + 1) * tf.exp(-b[2] * r) +
#              a[3] * (2. / (b[3] * r) + 1) * tf.exp(-b[3] * r) +
#              a[4] * (2. / (b[4] * r) + 1) * tf.exp(-b[4] * r))
#
#         dvdr = - (a[0] * (2 / (b[0] * rcut**2) + 2 / r + b[0]) * tf.exp(-b[0] * r) +
#                   a[1] * (2 / (b[1] * rcut**2) + 2 / r + b[1]) * tf.exp(-b[1] * r) +
#                   a[2] * (2 / (b[2] * rcut**2) + 2 / r + b[2]) * tf.exp(-b[2] * r) +
#                   a[3] * (2 / (b[3] * rcut**2) + 2 / r + b[3]) * tf.exp(-b[3] * r) +
#                   a[4] * (2 / (b[4] * rcut**2) + 2 / r + b[4]) * tf.exp(-b[4] * r))
#
#         v(r) - v(r_cut) - (r - r_cut) * dvdr(r_cut)
#
#         f(g) - f(g_cut) - (g - g_cut) * dfdg(g_cut)
#
#         return
#
#     return v
