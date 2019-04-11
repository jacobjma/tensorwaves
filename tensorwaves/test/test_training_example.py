import os

import numpy as np
import tensorflow as tf
from ase.io import read


# def test_training_example_stem():
#     dir = os.path.dirname(os.path.realpath(__file__))
#
#     atoms = read(dir + '/data/atoms00000.cif')
#     image = np.load(dir + '/data/image00000.npy')
#
#     example = TrainingExampleSTEM(atoms=atoms, image=image)
#
#     assert np.all(example.extent == np.array([20, 20], dtype=np.float32))
#     assert np.all(example.gpts == np.array([200, 200], dtype=np.int32))
#     assert np.all(example.sampling == np.array([.1, .1], dtype=np.float32))
#
#     example.resample(.05)
#
#     assert np.all(example.extent == np.array([20, 20], dtype=np.float32))
#     assert np.all(example.gpts == np.array([400, 400], dtype=np.int32))
#     assert np.all(example.sampling == np.array([.05, .05], dtype=np.float32))
#
#     example.cluster_and_classify(distance=.01, fingerprints=((42,), (16, 16), (16,)))
