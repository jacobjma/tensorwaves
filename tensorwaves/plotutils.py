import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show_image(image, ax=None, cmap='gray', **kwargs):
    if ax is None:
        ax = plt.subplot()

    mapable = ax.imshow(image.T, cmap=cmap, **kwargs)

    return ax, mapable


def add_colorbar(ax, mapable, position='right', size='5%', pad=0.05, **kwargs):
    if position in ['right', 'left']:
        orientation = 'vertical'
    elif position in ['top', 'bottom']:
        orientation = 'horizontal'

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    plt.colorbar(mapable, cax=cax, orientation=orientation, **kwargs)

    return cax
