import matplotlib.pyplot as plt
import numpy as np
from ase.data import chemical_symbols
from scipy.cluster.hierarchy import linkage, fcluster
from ase.symbols import Symbols


def generate_indices(labels):
    shape = labels.shape
    labels = labels.flatten()
    labels_order = labels.argsort()
    sorted_labels = labels[labels_order]
    indices = np.arange(0, len(labels) + 1)[labels_order]
    index = np.arange(1, np.max(labels) + 1)
    lo = np.searchsorted(sorted_labels, index, side='left')
    hi = np.searchsorted(sorted_labels, index, side='right')
    for i, (l, h) in enumerate(zip(lo, hi)):
        yield np.sort(indices[l:h])


def cluster_columns(atoms, tol=1e-6, longitudinal_ordering=False):
    xy = atoms.get_positions()[:, :2]
    z = atoms.get_positions()[:, 2]
    column_labels = fcluster(linkage(xy), tol, criterion='distance')

    n_columns = len(np.unique(column_labels))
    positions = np.zeros((n_columns, 2), dtype=np.float)
    labels = np.zeros(n_columns, dtype=np.int)
    column_types = []

    for i, indices in enumerate(generate_indices(column_labels)):
        numbers = atoms.get_atomic_numbers()[indices]

        if longitudinal_ordering:
            order = np.argsort(z[indices])
            numbers = numbers[order]
        else:
            numbers = np.sort(numbers)

        key = Symbols(numbers).get_chemical_formula()  # tuple(chemical_symbols[number] for number in numbers)

        positions[i] = np.mean(atoms.get_positions()[indices, :2], axis=0)

        try:
            labels[i] = column_types.index(key)

        except ValueError:
            column_types.append(key)

            labels[i] = len(column_types) - 1

        # except KeyError:
        #    column_labels[key] = []
        #    column_labels[key].append(indices)
    return positions, labels, column_types


def plot_columns(columns, atoms, marker='o', **kwargs):
    fig, ax = plt.subplots()

    for symbols, columns in columns.items():
        positions = np.zeros((len(columns), 2))
        for i, column in enumerate(columns):
            positions[i] = np.mean(atoms.get_positions()[column, :2], 0)
        plt.plot(*positions.T, label=symbols, marker=marker, linewidth=0, **kwargs)

    plt.legend()
    plt.axis('equal')

    return ax


def fwhm(profile):

    f = profile.numpy()

    amax = np.argmax(f)
    max_value = f[amax]

    f_shifted = np.abs(f - max_value / 2)

    left = np.argmin(f_shifted[:amax])

    x = np.linspace(0, 2, 3)
    y = f[[left - 1, left, left + 1]]

    p = np.polyfit(x, y, 1)
    left = left - 1 + (max_value / 2 - p[1]) / p[0]

    right = amax + np.argmin(f_shifted[amax:])
    y = f[[right - 1, right, right + 1]]

    p = np.polyfit(x, y, 1)
    right = right - 1 + (max_value / 2 - p[1]) / p[0]

    return (right - left) / len(f) * profile.extent[0]
