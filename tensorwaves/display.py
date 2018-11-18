import io
from collections import OrderedDict

import PIL.Image
import ipywidgets
import numpy as np
import traitlets
import bqplot

from tensorwaves import utils
from tensorwaves.ui import InterfaceBuilder
from tensorwaves.ctf import Aperture
from tensorwaves.bases import Node


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


class InteractiveDisplay(Node):
    description = traitlets.Unicode(default_value='Display')
    tensor_factory = traitlets.Instance(Aperture)

    display_space = traitlets.Unicode(default_value='Direct')
    mode = traitlets.Unicode(default_value='Magnitude')
    continuous_update = traitlets.Bool(default_value=False)

    x_scale = traitlets.Instance(bqplot.LinearScale)
    y_scale = traitlets.Instance(bqplot.LinearScale)
    x_axis = traitlets.Instance(bqplot.Axis)
    y_axis = traitlets.Instance(bqplot.Axis)
    figure = traitlets.Instance(bqplot.Figure)
    image = traitlets.Instance(bqplot.Image)

    i = traitlets.Int(default_value=0)

    @traitlets.validate('display_space')
    def _validate_space(self, proposal):
        if proposal['value'].lower() not in ('direct', 'fourier'):
            raise traitlets.TraitError()
        else:
            return proposal['value']

    @traitlets.validate('mode')
    def _validate_mode(self, proposal):
        if proposal['value'].lower() not in ('magnitude', 'real', 'imaginary'):
            raise traitlets.TraitError()
        else:
            return proposal['value']

    @traitlets.default('interface_builder')
    def _default_interface_builder(self):
        return InterfaceBuilder(self)

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

    @traitlets.default('image')
    def _default_image(self):
        image = self.get_image()
        return bqplot.Image(image=image, scales={'x': self.x_scale, 'y': self.y_scale})

    @traitlets.default('figure')
    def _default_figure(self):
        panzoom = bqplot.PanZoom(scales={'x': [self.x_scale], 'y': [self.y_scale]})
        return bqplot.Figure(marks=[self.image], axes=[self.x_axis, self.y_axis], interaction=panzoom,
                             min_aspect_ratio=1, max_aspect_ratio=1)

    def get_image(self):
        array = display_space_array(self.tensor_factory._tensor().numpy()[0].T,
                                    self.tensor_factory.space.space,
                                    self.display_space)
        array = convert_complex(array, self.mode)
        return array2image(array)

    def update_image(self, message=None):
        self.image.image = self.get_image()
        self.update_range()

    def update_range(self):
        x_range, y_range = display_space_range(self.tensor_factory.extent, self.tensor_factory.gpts,
                                               self.display_space)
        self.image.x = x_range
        self.image.y = y_range

    # def set_view

    def _link_update_image(self, include_upstream=True):
        for node, widgets in self.get_widgets(include_upstream=include_upstream).items():
            for name, widget in widgets.items():
                widget.observe(self.update_image, 'value')

    def _unlink_update_image(self, include_upstream=True):
        for node, widgets in self.get_widgets(include_upstream=include_upstream).items():
            for name, widget in widgets.items():
                widget.unobserve(self.update_image, 'value')

    @traitlets.observe('continuous_update')
    def _observe_continuous_update(self, change):
        if change['new']:
            self._link_update_image(include_upstream=True)
        else:
            self._unlink_update_image(include_upstream=True)

    def _build_widgets(self):
        widgets = OrderedDict()
        widgets['continuous_update'] = ipywidgets.ToggleButton(value=self.continuous_update,
                                                               description='Continuous update')
        widgets['display_space'] = ipywidgets.Dropdown(description='Display space', options=('Direct', 'Fourier'))
        widgets['mode'] = ipywidgets.Dropdown(description='Mode', options=('Magnitude', 'Real', 'Imaginary'))
        return widgets

    def _interface(self):
        manual_update = ipywidgets.Button(description='Update')
        manual_update.on_click(self.update_image)
        return ipywidgets.HBox([manual_update, super()._interface()])
