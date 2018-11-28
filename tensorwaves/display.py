import io
from collections import OrderedDict

import PIL.Image
import bqplot
import ipywidgets
import numpy as np
import tensorflow as tf
import traitlets

from tensorwaves.graph import upstream_traits, flatten_graph
from tensorwaves.widgets import link_widgets, layout_widgets


def array2image(array):
    image_array = ((array - array.min()) / (array.max() - array.min()) * 254 + 1).astype(np.uint8)
    image = PIL.Image.fromarray(image_array)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='png')
    return ipywidgets.Image(value=bytes_io.getvalue())


def convert_complex(array, mode='magnitude'):
    if mode.lower() == 'magnitude':
        return np.abs(array) ** 2
    elif mode.lower() == 'real':
        return np.real(array)
    elif mode.lower() == 'imaginary':
        return np.imag(array)
    else:
        raise RuntimeError()


def display_space_range(extent, gpts, display_space):
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


def display_space_array(array, space, display_space):
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


class Display(traitlets.HasTraits):
    display_space = traitlets.Unicode(default_value='Direct')

    x_scale = traitlets.Instance(bqplot.LinearScale)
    y_scale = traitlets.Instance(bqplot.LinearScale)
    x_axis = traitlets.Instance(bqplot.Axis)
    y_axis = traitlets.Instance(bqplot.Axis)
    figure = traitlets.Instance(bqplot.Figure)
    marks = traitlets.List()
    auto_update = traitlets.Bool(default_value=False)

    @traitlets.default('figure')
    def _default_figure(self):
        panzoom = bqplot.PanZoom(scales={'x': [self.x_scale], 'y': [self.y_scale]})
        axes = [self.x_axis, self.y_axis]
        self.update_labels()

        return bqplot.Figure(marks=self.marks, axes=axes,
                             interaction=panzoom,
                             fig_margin={'top': 40, 'bottom': 40, 'left': 40, 'right': 10},
                             min_aspect_ratio=1, max_aspect_ratio=2)

    @traitlets.default('x_scale')
    def _default_x_scale(self):
        return bqplot.LinearScale()

    @traitlets.default('y_scale')
    def _default_y_scale(self):
        return bqplot.LinearScale()

    @traitlets.default('x_axis')
    def _default_x_axis(self):
        return bqplot.Axis(scale=self.x_scale)

    @traitlets.default('y_axis')
    def _default_y_axis(self):
        return bqplot.Axis(scale=self.y_scale, orientation='vertical')

    @traitlets.validate('display_space')
    def _validate_space(self, proposal):
        if proposal['value'].lower() not in ('direct', 'fourier'):
            raise traitlets.TraitError()
        else:
            return proposal['value']

    def update_marks(self, message=None):
        return None

    def update_range(self, message=None):
        return None

    def update_labels(self, message=None):
        return None

    @traitlets.observe('auto_update')
    def _observe_auto_update(self, change):
        if change['new']:
            self._link_update()
        else:
            self._unlink_update()

    def _link_update(self):
        for node, traits in flatten_graph(upstream_traits(self)).items():
            for trait in traits:
                node.observe(self.update_marks, trait.name)

    def _unlink_update(self):
        for node, traits in flatten_graph(upstream_traits(self)).items():
            for trait in traits:
                node.unobserve(self.update_marks, trait.name)

    def widgets(self):
        widgets = OrderedDict()
        widgets['manual_update'] = ipywidgets.Button(description='Update')
        widgets['auto_update'] = ipywidgets.ToggleButton(value=self.auto_update, description='Auto-update')
        widgets['adjust_range'] = ipywidgets.Button(description='Adjust range')
        widgets['display_space'] = ipywidgets.Dropdown(description='Display space', options=('Direct', 'Fourier'))

        link_widgets(self, widgets)
        widgets['manual_update'].on_click(self.update_marks)
        widgets['adjust_range'].on_click(self.update_range)
        return widgets

    def show(self):
        def HBox(*pargs, **kwargs):
            box = ipywidgets.Box(*pargs, **kwargs)
            box.layout.display = 'flex'
            box.layout.align_items = 'flex-end'
            return box

        widgets = self.widgets()

        return HBox([self.figure, layout_widgets(widgets, 2, width=50)])


class LineDisplay(Display):
    description = traitlets.Unicode(default_value='LineDisplay')
    node = traitlets.Any()
    display_real = traitlets.Bool(default_value=True)
    display_imaginary = traitlets.Bool(default_value=True)
    display_magnitude = traitlets.Bool(default_value=True)
    continuous_update = traitlets.Bool(default_value=False)

    def get_data(self):
        return self.node._line_data()

    @traitlets.default('marks')
    def _default_marks(self):
        x, y = self.get_data()
        y = tf.imag(y)
        return [bqplot.Lines(x=x, y=y, scales={'x': self.x_scale, 'y': self.y_scale})]

    def update(self, message=None):
        x, y = self.get_data()
        y = tf.imag(y)
        self.marks[0].x = x
        self.marks[0].y = y


class ImageDisplay(Display):
    description = traitlets.Unicode(default_value='ImageDisplay')
    node = traitlets.Any()
    display_space = traitlets.Unicode(default_value='Direct')
    mode = traitlets.Unicode(default_value='Magnitude')
    marks = traitlets.List()

    i = traitlets.Int(default_value=0)

    @traitlets.validate('mode')
    def _validate_mode(self, proposal):
        if proposal['value'].lower() not in ('magnitude', 'real', 'imaginary'):
            raise traitlets.TraitError()
        else:
            return proposal['value']

    @traitlets.default('marks')
    def _default_marks(self):
        image = self.get_image()
        x_range, y_range = display_space_range(self.node.extent, self.node.gpts, self.display_space)
        image = bqplot.Image(image=image, scales={'x': self.x_scale, 'y': self.y_scale})
        image.x = x_range
        image.y = y_range
        return [image]

    def get_image(self):
        array = display_space_array(self.node._tensor().numpy()[0].T,
                                    self.node.space,
                                    self.display_space)
        array = convert_complex(array, self.mode)
        return array2image(array)

    def update_marks(self, message=None):
        self.update_labels()
        self.marks[0].image = self.get_image()
        self.update_range()

    def update_range(self, message=None):
        x_range, y_range = display_space_range(self.node.extent, self.node.gpts, self.display_space)
        self.marks[0].x = x_range
        self.marks[0].y = y_range

    def update_labels(self, message=None):
        if self.display_space.lower() == 'direct':
            self.x_axis.label = 'x [Angstrom]'
            self.y_axis.label = 'y [Angstrom]'
        elif self.display_space.lower() == 'fourier':
            self.x_axis.label = 'kx [1 / Angstrom]'
            self.y_axis.label = 'ky [1 / Angstrom]'

    def build_widgets(self):
        widgets = OrderedDict()
        widgets['mode'] = ipywidgets.Dropdown(description='Mode', options=('Magnitude', 'Real', 'Imaginary'))
        link_widgets(self, widgets)
        return widgets

    def interface(self):
        widgets = super().build_widgets()
        widgets.update(self.build_widgets())
        return layout_widgets(widgets, 2, width=50)


# def rgb2hex(rgb):
#     if len(rgb) > 0:
#         return list(np.apply_along_axis(lambda x: "#{:02x}{:02x}{:02x}".format(*x), 1, (rgb * 255).astype(np.int)))
#     else:
#         return []
#
#
# class ProjectedDisplay(Display):
#     potential = traitlets.Union([traitlets.Instance(Potential)])
#     plane = traitlets.Unicode(default_value='xy')
#     show_slice = traitlets.Bool(default_value=False)
#     slice = traitlets.Int(default_value=0, min=0)
#
#     @traitlets.validate('potential')
#     def _validate_potential(self, proposal):
#         self.traits()['slice'].max = proposal['value'].slices - 1
#         return proposal['value']
#
#     @property
#     def _source(self):
#         if self.show_slice:
#             return self.potential[self.slice]
#         else:
#             return self.potential
#
#     def _axes(self):
#         axes = ()
#         for axis in list(self.plane):
#             if axis == 'x': axes += (0,)
#             if axis == 'y': axes += (1,)
#             if axis == 'z': axes += (2,)
#         return axes
#
#     def _get_positions(self):
#         return self._source.positions[:, self._axes()].T
#
#     def _get_colors(self):
#         return rgb2hex(cpk_colors[self._source.atomic_numbers])
#
#     def _get_atomic_sizes(self):
#         return covalent_radii[self._source.atomic_numbers]
#
#     def _get_box(self):
#         axes = self._axes()
#         box = np.zeros((2, 5))
#         box[0, :] = self._source.origin[axes[0]]
#         box[1, :] = self._source.origin[axes[1]]
#         box[0, 2:4] += self._source.box[axes[0]]
#         box[1, 1:3] += self._source.box[axes[1]]
#         return box
#
#     @traitlets.default('marks')
#     def _default_marks(self):
#         x, y = self._get_positions()
#         atoms = bqplot.Scatter(x=x, y=y, scales={'x': self.x_scale, 'y': self.y_scale},
#                                colors=self._get_colors(), stroke_width=1.5, stroke='black')
#
#         x, y = self._get_box()
#         box = bqplot.Lines(x=x, y=y, scales={'x': self.x_scale, 'y': self.y_scale}, colors=['black'])
#         return [atoms, box]
#
#     def update_marks(self, message=None):
#         x, y = self._get_positions()
#         self.marks[0].x = x
#         self.marks[0].y = y
#         self.marks[0].colors = self._get_colors()
#
#         x, y = self._get_box()
#         self.marks[1].x = x
#         self.marks[1].y = y
#
#     @traitlets.observe('slice')
#     def _observe_slice(self, change):
#         if isinstance(self.potential, PotentialSlice):
#             self.potential.index = change['new']
#
#     def widgets(self):
#         widgets = OrderedDict()
#         widgets['show_slice'] = ipywidgets.ToggleButton(value=self.show_slice, description='Show slice')
#         widgets['slice'] = ipywidgets.BoundedIntText(description='Slice Number', value=self.slice, min=0,
#                                                      max=self.traits()['slice'].max)
#         widgets['plane'] = ipywidgets.Dropdown(options=['xy', 'xz', 'yz'], value=self.plane, description='Plane')
#
#         link_widgets(self, widgets)
#         parent_widgets = super().widgets()
#         parent_widgets.update(widgets)
#         parent_widgets['display_space'].disabled = True
#
#         return parent_widgets
