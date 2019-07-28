import os

import numpy as np


def generate_indices(labels):
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(1, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


def walk_dir(path, ending):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-len(ending):] == ending:
                files.append(os.path.join(r, file))

    return files
