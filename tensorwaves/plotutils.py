import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.gridspec as gridspec


def convert_complex(array, mode='magnitude'):
    if mode.lower()[:3] == 'mag':
        return np.abs(array) ** 2
    elif mode.lower()[:3] == 'int':
        return np.abs(array) ** 2
    elif mode.lower()[:3] == 'rea':
        return np.real(array)
    elif mode.lower()[:3] == 'ima':
        return np.imag(array)
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


def show_array(array, extent, space, display_space='direct', mode='magnitude', colorbar=True, vmin=None,
               vmax=None, num_cols=4, **kwargs):
    num_images = len(array)
    num_cols = min(num_cols, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    x_range, y_range = plot_range(extent, array.shape[1:], display_space)

    image_aspect = (y_range[-1] - y_range[0]) / (x_range[-1] - x_range[0])
    fig_aspect = num_rows / num_cols * image_aspect

    fig = plt.figure(figsize=(3 * num_cols + .5, 3 * fig_aspect * num_cols))
    gs = gridspec.GridSpec(num_rows + 1, num_cols + 2,
                           height_ratios=num_rows * [1] + [0.0001],
                           width_ratios=[0.0001] + num_cols * [1] + [.05])

    images = convert_complex(plot_array(array, space, display_space), mode=mode)

    if vmin is None:
        vmin = np.min(images)

    if vmax is None:
        vmax = np.max(images)

    for i in range(len(images)):
        ax = plt.subplot(gs[i // num_cols, (i % num_cols) + 1])

        mapable = ax.imshow(images[i].T, extent=x_range + y_range, origin='lower', vmin=vmin, vmax=vmax)

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
                                             height_ratios=[(num_rows - 1)/2, 1, (num_rows - 1)/2])
    ax = plt.subplot(right[1, 0])
    plt.colorbar(mapable, cax=ax, orientation='vertical', label='', )

    #gs.tight_layout(fig)


def show_line(x, y, mode, ax=None, **kwargs):
    if ax is None:
        ax = plt.subplot()

    ax.plot(x, convert_complex(y, mode=mode), **kwargs)


def add_colorbar(ax, mapable, position='right', size='5%', pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'
    else:
        raise RuntimeError()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    plt.colorbar(mapable, cax=cax, orientation=orientation, **kwargs)

    return cax
