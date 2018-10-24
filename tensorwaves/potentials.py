import numpy as np
import tensorflow as tf
from bases import FactoryBase, TensorBase
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq
from potential_parametrizations import potential_dict

def log_grid(start, stop, num):
    dt = np.log(stop / start) / (num - 1)
    return start * np.exp(dt * np.linspace(0, num - 1, num))


def find_cutoff(func, min_value, xmin=1e-16, xmax=1000):
    return brentq(lambda x: func(x) - min_value, xmin, xmax)


def potential_spline_coeff(r_min, v_min, n_nodes, v, dvdr):
    r_cut = find_cutoff(v, v_min)

    r = log_grid(r_min, r_cut, n_nodes)

    v = v(r) - v(r_cut) - (r - r_cut) * dvdr(r_cut)

    bc_left, bc_right = [(1, dvdr(r_min))], [(1, 0.)]

    return make_interp_spline(r, v, bc_type=(bc_left, bc_right))


class Potential(FactoryBase):

    def __init__(self, atoms, gpts=None, sampling=None):
        super().__init__(gpts=gpts, sampling=sampling)

        if atoms is None:
            self._atoms = None
        else:
            self.atoms = atoms

    @property
    def extent(self):
        return np.diag(self._atoms.get_cell())

    @extent.setter
    def extent(self, _):
        raise RuntimeError()


class ProjectedPotentialChunk(object):

    def __init__(self, array, axes):
        self._array = array
        self._axes = axes

    axes = property(lambda self: self._axes)
    array = property(lambda self: self._array)


class CustomPotential(TensorBase):

    def __init__(self, array, axes):
        if len(array.shape) != 3:
            raise ValueError()

        self._array = tf.constant(array, dtype=tf.float32)

        super(CustomPotential, self).__init__(axes)

    array = property(lambda self: self._array)

    def project_potential_chunk(self, n_slice=None, h_slice=None):
        axes = Axes(self.axes.lst[:2] + [LinearAxis(l=self.axes.lz, n=n_slice, h=h_slice)])

        array = tf.reduce_sum(tf.reshape(self._array, (self.axes.nx, self.axes.ny, axes.nz, -1)),
                              axis=3) * self._axes.hz

        return ProjectedPotentialChunk(array, axes)

    def chunk_generator(self):
        yield self.project_potential_chunk(self._box)


class ParamPotential(Potential):

    def __init__(self):
        super(ParamPotential, self).__init__()
