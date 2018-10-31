import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ase.data.colors import cpk_colors
from ase.data import covalent_radii
from ase import Atoms

from tensorwaves.potentials import Potential, PotentialSlice
from tensorwaves.atoms import SlicedAtoms

def show_image(image, ax=None, cmap='gray', **kwargs):
    if ax is None:
        ax = plt.subplot()

    mapable = ax.imshow(image.T, cmap=cmap, **kwargs)

    return ax, mapable


def axis2idx(axis):
    if axis == 'x':
        return 0
    if axis == 'y':
        return 1
    if axis == 'z':
        return 2


def show_atoms(atoms, plane='xy', ax=None, scale=1, linewidth=2, edgecolor='k', show_margin=True, **kwargs):
    if ax is None:
        ax = plt.subplot()

    if isinstance(atoms, Potential):
        atoms = atoms._sliced_atoms
    elif isinstance(atoms, Atoms):
        atoms = SlicedAtoms(atoms)

    axes_idx = [axis2idx(axis) for axis in list(plane)]

    c = cpk_colors[atoms.atomic_numbers.numpy()]
    s = scale * covalent_radii[atoms.atomic_numbers.numpy()]

    positions = atoms.positions.numpy()[:, axes_idx]
    ax.scatter(*positions.T, c=c, s=s, linewidth=linewidth, edgecolor=edgecolor, **kwargs)
    #if show_margin:
    #    ax.scatter(*positions.T[len(atoms):], c=c, s=s, linewidth=linewidth, edgecolor=edgecolor, **kwargs)

    ax.axis('equal')

    a = [0., 0., atoms.entrance_plane][axes_idx[0]]
    b = [0., 0., atoms.entrance_plane][axes_idx[1]]
    la = atoms.box[axes_idx[0]]
    lb = atoms.box[axes_idx[1]]
    ax.plot([a, a, a + la, a + la, a],
            [b, b + lb, b + lb, b, b], 'k',
            linewidth=1.5)

    if axes_idx[1] == 2:
        ax.invert_yaxis()

    return ax


def add_colorbar(ax, mapable, position='right', size='5%', pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    plt.colorbar(mapable, cax=cax, orientation=orientation, **kwargs)

    return cax
