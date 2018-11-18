import IPython.display
import bokeh.io
import bokeh.layouts
import bokeh.models
import ipywidgets as widgets
import tensorflow as tf
import traitlets as traits
from ase.data import covalent_radii
from ase.data.colors import cpk_colors
from bokeh.plotting import figure

from tensorwaves import utils


def display_space_tensor(tensor, space, display_space):
    if display_space is 'direct':
        if space == 'fourier':
            tensor = utils.fft_shift(tf.ifft2d(tf.cast(tensor, tf.complex64)), (0, 1))

    elif display_space is 'fourier':
        if space is 'fourier':
            tensor = utils.fft_shift(tensor, (0, 1))
        else:
            tensor = utils.fft_shift(tf.fft2d(tf.cast(tensor, tf.complex64)), (0, 1))

    else:
        raise RuntimeError()

    return tensor


def display_space_range(extent, shape, display_space):
    if display_space == 'direct':
        x_range = [0, extent[0]]
        y_range = [0, extent[1]]

    elif display_space == 'fourier':
        extent = [1 / extent[0] * shape[0], 1 / extent[1] * shape[1]]
        x_range = [-extent[0] / 2, extent[0] / 2]
        y_range = [-extent[1] / 2, extent[1] / 2]

    else:
        raise RuntimeError()

    return x_range, y_range


def display_space_labels(display_space):
    if display_space == 'direct':
        x_label = 'x [Angstrom]'
        y_label = 'y [Angstrom]'

    elif display_space == 'fourier':
        x_label = 'kx [1/Angstrom]'
        y_label = 'ky [1/Angstrom]'
    else:
        raise RuntimeError()

    return x_label, y_label


class InteractiveDisplay(traits.HasTraits):
    display_space = traits.Unicode(default_value='direct')
    mode = traits.Unicode(default_value='magnitude')
    update = traits.Unicode(default_value='manual')
    plot_width = traits.Int(default_value=400)
    plot_height = traits.Int(default_value=400)
    i = traits.Int(default_value=0)

    @traits.validate('display_space')
    def _validate_space(self, proposal):
        if proposal['value'].lower() not in ('direct', 'fourier'):
            raise traits.TraitError()
        else:
            return proposal['value']

    @traits.validate('mode')
    def _validate_mode(self, proposal):
        if proposal['value'].lower() not in ('magnitude', 'real', 'imaginary'):
            raise traits.TraitError()
        else:
            return proposal['value']

    def new_figure(self):
        x_range, y_range = display_space_range(self.instance.factory_base.extent, self.instance.factory_base.gpts,
                                               self.display_space)

        self.figure = figure(plot_width=self.plot_width, plot_height=self.plot_height, x_range=x_range, y_range=y_range)

    def get_image_data(self):
        tensor = display_space_tensor(self.instance._tensor()[self.i],
                                      self.instance.factory_base.space,
                                      self.display_space)
        return utils.convert_complex(tensor, self.mode).numpy().T

    def new_image(self):
        image_data = self.get_image_data()

        x_range, y_range = display_space_range(self.instance.factory_base.extent, self.instance.factory_base.gpts,
                                               self.display_space)

        self.image = self.figure.image([image_data], x=x_range[0], y=y_range[0],
                                       dw=x_range[1] - x_range[0],
                                       dh=y_range[1] - y_range[0], palette='Viridis256')

    def update_image(self, _):
        self.image.data_source.data['image'] = [self.get_image_data()]
        x_range, y_range = display_space_range(self.instance.factory_base.extent, self.instance.factory_base.gpts,
                                               self.display_space)

        self.image.glyph.x = x_range[0]
        self.image.glyph.y = y_range[0]
        self.image.glyph.dw = x_range[1] - x_range[0]
        self.image.glyph.dh = y_range[1] - y_range[0]

        bokeh.io.push_notebook()

    def update_range(self, _):
        x_range, y_range = display_space_range(self.instance.factory_base.extent, self.instance.factory_base.gpts,
                                               self.display_space)

        self.figure.x_range.start = x_range[0]
        self.figure.x_range.end = x_range[1]
        self.figure.y_range.start = y_range[0]
        self.figure.y_range.end = y_range[1]
        self.figure.x_range.reset_start = x_range[0]
        self.figure.x_range.reset_end = x_range[1]
        self.figure.y_range.reset_start = y_range[0]
        self.figure.y_range.reset_end = y_range[1]

        bokeh.io.push_notebook()

    def display(self, instance):

        self.instance = instance
        self.new_figure()
        self.new_image()
        self.update_range(None)

        bokeh.io.show(self.figure, notebook_handle=True)

        self.observe(self.update_image, 'mode')
        self.observe(self.update_image, 'display_space')
        self.observe(self.update_range, 'display_space')

        if self.update is 'manual':
            update = widgets.Button(description='Update image')
            update.on_click(self.update_image)
            IPython.display.display(update)
        else:
            self.instance.grid.x_grid.observe(self.update_image, ('gpts', 'sampling', 'extent'))
            self.instance.grid.y_grid.observe(self.update_image, ('gpts', 'sampling', 'extent'))
            self.instance.accelerator.observe(self.update_image, 'energy')
            self.instance.calculation_space.observe(self.update_image, 'space')
            self.observe(self.update_image, 'display_space')
            self.observe(self.update_image, 'mode')

    def interact(self):
        mode = widgets.Dropdown(options=('magnitude', 'real', 'imaginary'), description='mode:', disabled=False)
        space = widgets.Dropdown(options=('fourier', 'direct'), description='display space:', disabled=False)
        reset = widgets.Button(description="Fit to range")
        traits.link((self, 'mode'), (mode, 'value'))
        traits.link((self, 'display_space'), (space, 'value'))
        reset.on_click(self.update_range)

        return widgets.HBox((mode, space, reset))


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

    axes_idx = [axis2idx(axis) for axis in list(plane)]

    positions = atoms.positions[:, axes_idx]
    atomic_numbers = atoms.atomic_numbers

    c = cpk_colors[atomic_numbers]
    s = scale * covalent_radii[atomic_numbers]

    positions = positions
    ax.scatter(*positions.T, c=c, s=s, linewidth=linewidth, edgecolor=edgecolor, **kwargs)
    # if show_margin:
    #    ax.scatter(*positions.T[len(atoms):], c=c, s=s, linewidth=linewidth, edgecolor=edgecolor, **kwargs)

    ax.axis('equal')

    a = atoms.origin[axes_idx[0]]
    b = atoms.origin[axes_idx[1]]
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
