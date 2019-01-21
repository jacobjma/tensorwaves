import sys
sys.path.append("/Users/jacobmadsen/PycharmProjects/tensorwaves")

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

from tensorwaves.waves import PrismWaves
from tensorwaves.potentials import Potential
from tensorwaves.scan import GridScan
from tensorwaves.detect import WaveDetector

from ase.io import read

atoms = read('gra66_Si-C3.POSCAR')

L = 38
tol = 1.8

repeated_atoms = atoms * (3,3,1)
cell = repeated_atoms.get_cell()
repeated_atoms.set_cell([L, L, cell[2,2]])
repeated_atoms.center()

del repeated_atoms[repeated_atoms.get_positions()[:,0]<tol]
del repeated_atoms[repeated_atoms.get_positions()[:,0]>L-tol]

del repeated_atoms[repeated_atoms.get_positions()[:,1]<tol]
del repeated_atoms[repeated_atoms.get_positions()[:,1]>L-tol]

potential = Potential(atoms = repeated_atoms, gpts=(64,64))

prism = PrismWaves(energy=80e3, cutoff=.035, interpolation=1)

prism.grid.match(potential.grid)

S = prism.multislice(potential, in_place=True)

S.aperture.radius=.034
S.aperture.rolloff=.001

detector = WaveDetector()

center = np.array([L,L]) / 2
extent = np.diag(atoms.get_cell())[:2]

scan = GridScan(scanable=S, detectors=detector, num_positions=10, start=center - extent / 2,
                end=center + extent / 2)

scan.scan()

cube = np.vstack([wave.numpy() for wave in scan._data[detector]])

np.save('cube.npy', cube)