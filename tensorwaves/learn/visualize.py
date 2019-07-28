import io

import PIL
import ipywidgets as widgets
import numpy as np
import numbers

def normalize_range(array):
    return ((array - array.min()) / (array.max() - array.min()))


def array_as_bytes(array):
    image = PIL.Image.fromarray(array)
    bytes_io = io.BytesIO()
    image.save(bytes_io, format='png')
    return bytes_io.getvalue()


class ImageDisplays(object):

    def __init__(self, n=1, m=1, width=300, height=300):
        self._current_index = 0

        self._widgets = []
        for i in range(m):
            self._widgets.append([])
            for j in range(n):
                self._widgets[-1].append(widgets.Image(width=width, height=height))

    def widget(self):
        rows = []
        for row in self._widgets:
            rows.append(widgets.HBox(row))
        return widgets.VBox(rows)

    def update(self, n, m, image):
        try:
            self._widgets[m][n].value = image
        except:
            self._widgets[m][n].value = array_as_bytes(image.astype(np.uint8))


class ValueDisplays(object):

    def __init__(self, keys):
        self._widgets = {}
        for key in keys:
            self._widgets[key] = widgets.HTML('{} :'.format(key))

    def update(self, key, value):

        if isinstance(value, bool):
            value = '{}'.format(value)

        elif not isinstance(value, numbers.Integral):
            value = '{:.2f}'.format(value)

        self._widgets[key].value = '{}: {}'.format(key, value)

    def widget(self):
        return widgets.VBox(list(self._widgets.values()))


class FrameControl(object):

    def __init__(self, funcs=None, **kwargs):
        self._int_slider = widgets.IntSlider(**kwargs)

        if funcs is None:
            self._funcs = []
        else:
            self._funcs = funcs

        def on_frame_change(change):
            for func in self._funcs:
                func(change['new'])

        self._int_slider.observe(on_frame_change, names='value')

        on_frame_change({'new': 0})

    @property
    def int_slider(self):
        return self._int_slider

    @property
    def next_button(self):
        next_button = widgets.Button(description='Next')
        next_button.on_click(self.next_frame)
        return next_button

    @property
    def previous_button(self):
        previous_button = widgets.Button(description='Previous')
        previous_button.on_click(self.previous_frame)
        return previous_button

    def next_frame(self, _=None):
        self.int_slider.value = (self.int_slider.value + 1) % (self.int_slider.max + 1)

    def previous_frame(self, _=None):
        self.int_slider.value = (self.int_slider.value - 1) % (self.int_slider.max + 1)

    def widget(self):
        return widgets.VBox([widgets.HBox([self.previous_button, self.next_button]), self.int_slider])
