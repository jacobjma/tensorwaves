import io
import time

import PIL.Image
import bqplot
import ipywidgets
import numpy as np
from bqplot import LinearScale, Axis, Figure, PanZoom

from tensorwaves.bases import HasData, notifying_property
from tensorwaves.plotutils import convert_complex, plot_array, plot_range


def link_widget(o, widget, property_name):
    def callback(change):
        setattr(o, property_name, change['new'])

    widget.observe(callback, 'value')
    return callback


def link_widget_component(widget, o, property_name, component):
    def callback(change):
        value = getattr(o, property_name).copy()
        value[component] = change['new']
        setattr(o, property_name, value)

    widget.observe(callback, 'value')
    return callback


class InteractiveDisplay(HasData):

    def __init__(self, showable, space='direct', mode='magnitude', auto_update=False, margin=None):
        self._showable = showable
        self._space = space
        self._mode = mode
        self._auto_update = auto_update
        self._updating = False

        self.scales = {'x': LinearScale(), 'y': LinearScale()}
        self.axes = {'x': Axis(scale=self.scales['x']),
                     'y': Axis(scale=self.scales['y'], orientation='vertical')}

        panzoom = PanZoom(scales={'x': [self.scales['x']], 'y': [self.scales['y']]})

        if margin is None:
            margin = {'top': 0, 'bottom': 40, 'left': 80, 'right': 10}

        self.figure = Figure(marks=[], axes=list(self.axes.values()), interaction=panzoom, fig_margin=margin)

        self._last_update = None

        HasData.__init__(self, save_data=True)

    space = notifying_property('_space')
    mode = notifying_property('_mode')
    auto_update = notifying_property('_auto_update')

    def notify(self, observable, message):
        if self.auto_update & message['change'] & (not self._updating):
            self._updating = True
            if observable is not self:
                self.update_data()
            self.update()
            self._updating = False

    def update(self):
        raise NotImplementedError()

    def update_data(self):
        raise NotImplementedError()

    def update_marks(self):
        raise NotImplementedError()

    def update_coordinates(self):
        raise NotImplementedError()

    def update_labels(self):
        raise NotImplementedError()

    def update_info(self):
        raise NotImplementedError()

    def show(self):
        return self.figure


class ImageDisplay(InteractiveDisplay):

    def __init__(self, showable, space='direct', mode='magnitude', color_scale='linear', auto_update=False,
                 margin=None):
        InteractiveDisplay.__init__(self, showable=showable, space=space, auto_update=auto_update, margin=margin,
                                    mode=mode)
        self._color_scale = color_scale
        self.update_data()

        self.figure.marks = [bqplot.Image(image=self._get_image(), scales=self.scales)]
        self.update_coordinates()
        self.update_labels()

        for observed in showable._observing:
            observed.register_observer(self)

        self._showable.register_observer(self)
        self.register_observer(self)

    color_scale = notifying_property('_color_scale')

    def update(self):
        self.update_marks()
        self.update_coordinates()
        self.update_labels()
        self.update_info()

    def update_data(self):
        t = time.time()
        self._data = self._showable.get_showable_tensor()
        self._last_update = time.time() - t

    def update_marks(self, message=None):
        self.figure.marks[0].image = self._get_image()

    def update_coordinates(self, message=None):
        x_range, y_range = plot_range(self._data.grid.extent, self._data.grid.gpts, self.space)
        self.figure.marks[0].x = x_range
        self.figure.marks[0].y = y_range

    def update_range(self, message=None):
        x_range, y_range = plot_range(self._data.grid.extent, self._data.grid.gpts, self.space)
        self.scales['x'].min, self.scales['x'].max = map(float, x_range)
        self.scales['y'].min, self.scales['y'].max = map(float, y_range)

    def update_labels(self, message=None):
        if self.space.lower() == 'direct':
            self.axes['x'].label = 'x [Angstrom]'
            self.axes['y'].label = 'y [Angstrom]'
        elif self.space.lower() == 'fourier':
            self.axes['x'].label = 'kx [1 / Angstrom]'
            self.axes['y'].label = 'ky [1 / Angstrom]'

    def update_info(self):
        self._info_widget.value = 'Last update: {:.5f} s'.format(self._last_update)

    def _get_image(self):
        array = self._data.numpy()[0]
        array = convert_complex(plot_array(array, self._data.space, self._space), mode=self._mode)

        if self._color_scale == 'log':
            sign = np.sign(array)
            array = sign * np.log(1 + np.abs(array))

        array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        image = PIL.Image.fromarray(np.flipud(array.T))
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='png')
        return ipywidgets.Image(value=bytes_io.getvalue())

    def show(self):
        update_button = ipywidgets.Button(description='Update')

        def on_click(_):
            self.update_data()
            self.update()

        update_button.on_click(on_click)

        auto_update_button = ipywidgets.ToggleButton(value=self.auto_update, description='Auto-update')
        link_widget(self, auto_update_button, 'auto_update')

        reset_button = ipywidgets.Button(description='Reset range')
        reset_button.on_click(self.update_range)

        zoom_button = ipywidgets.Button(description='Zoom')
        # zoom_button.on_click(self.update_range)

        space_widget = ipywidgets.Dropdown(description='Display space', options=('direct', 'fourier'), value=self.space,
                                           style={'description_width': '100px'})
        link_widget(self, space_widget, 'space')

        mode_widget = ipywidgets.Dropdown(description='Mode', options=('magnitude', 'real', 'imaginary'),
                                          value=self.mode, style={'description_width': '100px'})
        link_widget(self, mode_widget, 'mode')

        color_scale_widget = ipywidgets.Dropdown(description='Scale', options=('log', 'linear'),
                                                 value=self._color_scale, style={'description_width': '100px'})
        link_widget(self, color_scale_widget, 'color_scale')

        self._info_widget = ipywidgets.HTML()
        self.update_info()

        widget_box = ipywidgets.VBox([ipywidgets.HBox([update_button, auto_update_button]),
                                      ipywidgets.HBox([reset_button, zoom_button]),
                                      ipywidgets.HBox([space_widget]),
                                      ipywidgets.HBox([mode_widget]),
                                      ipywidgets.HBox([color_scale_widget]),
                                      ipywidgets.HBox([self._info_widget])])

        box = ipywidgets.Box([self.figure, widget_box],
                             layout=ipywidgets.Layout(display='flex', align_items='flex-start'))
        return box
