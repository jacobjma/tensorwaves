import sys
sys.path.append("/Users/jacobmadsen/PycharmProjects/tensorwaves")

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

from tensorwaves.waves import PrismWaves
from tensorwaves.detect import RingDetector
from tensorwaves.learn import TrainingSetSTEM, TrainingExampleSTEM
from tensorwaves.utils import ProgressBar, ProgressTracker
from tensorwaves.model.graphene import sheet, apply_random_strain


def random_graphene_Si_sheet(box=(8, 8, 4), edge_tol=.5, min_replace=1, max_replace=3):
    rotation = np.random.rand() * 60

    atoms = sheet(box=box, rotation=rotation, edge_tol=edge_tol)

    atoms = apply_random_strain(atoms, 0, amplitude=.2, power=-4)
    atoms = apply_random_strain(atoms, 1, amplitude=.2, power=-4)
    atoms = apply_random_strain(atoms, 2, amplitude=.2, power=-4)

    atoms.wrap()

    for index in np.random.choice(len(atoms), np.random.randint(min_replace, max_replace)):
        atoms[index].symbol = 'Si'

    return atoms


num_examples = 2
num_positions = (48, 48)

bar = ProgressBar(num_iter=num_examples, description='Example #')
tracker = ProgressTracker()
tracker.add_bar(bar)

examples = []

for i in range(num_examples):
    bar.update(i)

    atoms = random_graphene_Si_sheet()

    example = TrainingExampleSTEM(atoms=atoms, num_positions=num_positions)

    prism = PrismWaves(interpolation=1, sampling=.05, energy=80e3, cutoff=.025, )

    detector = RingDetector(inner=.05, outer=.25, rolloff=.02)

    example.classify(0., [(6,), (14,)])

    example.create_truth(.001, [(6,), (14,)], 3)

    example.create_image(prism=prism, detector=detector, tracker=tracker)

    examples.append(example)

training_set = TrainingSetSTEM(examples=examples)
training_set.save('test')