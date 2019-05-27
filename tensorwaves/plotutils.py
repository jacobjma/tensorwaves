from colorsys import hls_to_rgb

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def domain_coloring(z, fade_to_white=False, saturation=1, k=.5):
    h = (np.angle(z) + np.pi) / (2 * np.pi) + 0.5
    if fade_to_white:
        l = k ** np.abs(z)
    else:
        l = 1 - k ** np.abs(z)
    c = np.vectorize(hls_to_rgb)(h, l, saturation)
    return np.array(c).T


def convert_complex(array, mode='magnitude'):
    if mode.lower()[:3] == 'mag':
        return np.abs(array)
    elif mode.lower()[:3] == 'int':
        return np.abs(array) ** 2
    elif mode.lower()[:3] == 'rea':
        return np.real(array)
    elif mode.lower()[:3] == 'ima':
        return np.imag(array)
    elif mode.lower()[:3] == 'pha':
        return np.angle(array)
    else:
        raise RuntimeError()


def plot_range(extent, gpts, display_space):
    if display_space.lower() == 'direct':
        x_range = [0, extent[0]]
        y_range = [0, extent[1]]

    elif display_space.lower() == 'fourier':
        extent = [1 / extent[0] * gpts[0], 1 / extent[1] * gpts[1]]
        x_range = [-extent[0] / 2, extent[0] / 2]
        y_range = [-extent[1] / 2, extent[1] / 2]

    else:
        raise RuntimeError()

    return x_range, y_range


def plot_array(array, space, display_space):
    if display_space.lower() == 'direct':
        if space.lower() == 'fourier':
            array = np.fft.fftshift(np.fft.ifft2(array))
    elif display_space.lower() == 'fourier':
        if space.lower() == 'fourier':
            array = np.fft.fftshift(array)
        else:
            array = np.fft.fftshift(np.fft.fft2(array))
    else:
        raise RuntimeError()

    return array


def plot_labels(display_space):
    if display_space.lower() == 'direct':
        labels = ['x [Angstrom]', 'y [Angstrom]']
    elif display_space.lower() == 'fourier':
        labels = ['kx [1 / Angstrom]', 'ky [1 / Angstrom]']
    else:
        raise RuntimeError()
    return labels


def show_array(array, extent, space, display_space='direct', mode='intensity', scale='linear', colorbar=True, vmin=None,
               vmax=None, num_cols=4, fig_scale=1, tile=(1, 1), **kwargs):
    num_images = len(array)
    num_cols = min(num_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    extent = extent * tile
    array = np.tile(array, (1,) + tile)

    x_range, y_range = plot_range(extent, array.shape[1:], display_space)

    image_aspect = (y_range[-1] - y_range[0]) / (x_range[-1] - x_range[0])
    fig_aspect = num_rows / num_cols * image_aspect

    fig = plt.figure(figsize=(fig_scale * 3 * num_cols + .5, fig_scale * 3 * fig_aspect * num_cols))
    gs = gridspec.GridSpec(num_rows + 1, num_cols + 2,
                           height_ratios=num_rows * [1] + [0.0001],
                           width_ratios=[0.0001] + num_cols * [1] + [.05])

    images = convert_complex(plot_array(array, space, display_space), mode=mode)

    if scale == 'log':
        images = np.log(1 + images)

    if vmin is None:
        vmin = np.min(images)

    if vmax is None:
        vmax = np.max(images)

    for i in range(len(images)):
        ax = plt.subplot(gs[i // num_cols, (i % num_cols) + 1])

        mapable = ax.imshow(images[i].T, extent=x_range + y_range, origin='lower', vmin=vmin, vmax=vmax, **kwargs)

        # if i % num_cols:
        #    ax.set_yticks([])

        # if (i // num_cols) != (num_rows - 1):
        #    ax.set_xticks([])

    labels = plot_labels(display_space)

    bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[-1, 1:-1])
    ax = plt.subplot(bottom[0, 0], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(labels[0])

    left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:-1, 0])
    ax = plt.subplot(left[0, 0], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(labels[1])

    right = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:-1, -1],
                                             height_ratios=[(num_rows - 1) / 2, 1, (num_rows - 1) / 2])
    ax = plt.subplot(right[1, 0])
    plt.colorbar(mapable, cax=ax, orientation='vertical', label='', )

    return ax
    # gs.tight_layout(fig)


def show_line(x, y, mode, ax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()

    if np.any(np.iscomplex(y)):
        y = convert_complex(y, mode=mode)

    ax.plot(x, y, **kwargs)


# def add_colorbar(ax, mapable, position='right', size='5%', pad=0.05, **kwargs):
#     if position in ['right', 'left']:
#         orientation = 'vertical'
#     elif position in ['top', 'bottom']:
#         orientation = 'horizontal'
#     else:
#         raise RuntimeError()
#
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes(position, size=size, pad=pad)
#     plt.colorbar(mapable, cax=cax, orientation=orientation, **kwargs)
#
#     return cax


def plane2axes(plane):
    axes = ()
    for axis in list(plane):
        if axis == 'x': axes += (0,)
        if axis == 'y': axes += (1,)
        if axis == 'z': axes += (2,)
    return axes


def add_colorbar(ax, cmap, vmin, vmax, n_ticks=5, ticks=None, position='bottom', size='5%', pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    if ticks is None:
        ticks = np.linspace(vmin, vmax, n_ticks)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return plt.colorbar(sm, cax=cax, orientation=orientation, ticks=ticks, **kwargs)


def plot_atoms(atoms, plane='xy', ax=None, scale=1, fig_scale=1):
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    origin = np.array([0., 0., 0.])
    box = np.diag(atoms.get_cell())
    return _plot_atoms(positions=positions, numbers=numbers, plane=plane, origin=origin, box=box, scale=scale, ax=ax,
                       fig_scale=fig_scale)


def _plot_atoms(positions, numbers, plane, origin, box, scale=1, ax=None, colors=None, fig_scale=1):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8 * fig_scale, 6 * fig_scale))

    axes = plane2axes(plane)
    edges = np.zeros((2, 5))
    edges[0, :] += origin[axes[0]]
    edges[1, :] += origin[axes[1]]
    edges[0, 2:4] += np.array([box[0], box[1], box[2]])[axes[0]]
    edges[1, 1:3] += np.array([box[0], box[1], box[2]])[axes[1]]

    ax.plot(edges[0, :], edges[1, :], 'k-')

    if len(positions) > 0:
        positions = positions[:, axes]
        if colors is None:
            colors = cpk_colors[numbers]
        sizes = covalent_radii[numbers]

        ax.scatter(*positions.T, c=colors, s=scale * sizes * 100)
        ax.axis('equal')