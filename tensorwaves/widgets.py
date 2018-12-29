from ipywidgets import Layout, Label
from ipywidgets import BoundedFloatText, BoundedIntText, FloatText, VBox, HBox, Accordion
from collections import OrderedDict


def link_widgets(o, widgets):
    for name in widgets.keys():
        def get_observer(name):
            def observer(message):
                setattr(o, name, message['new'])

            return observer

        observer = get_observer(name)
        widgets[name].observe(observer, 'value')


class HasWidgets(object):

    def __init__(self, node=None):
        self._widgets = None
        self._node = node

    @property
    def widgets(self):
        if self._widgets is None:
            self._create_widgets()
        return self._widgets

    @property
    def node(self):
        return self._node

    def interface(self):
        return None

    def _create_widgets(self):
        return None

    def _add_callback(self, callback):
        # TODO: implement as observer pattern
        for widget in self.widgets.values():
            widget.observe(callback, 'value')

    def _link_widgets(self):
        link_widgets(self.node, self.widgets)

    def show(self, include_upstream=False):
        if include_upstream:
            rows = []
            for node in self.node.upstream_nodes():
                row = node.widgets._show()
                if row is not None:
                    rows.append(row)
            return VBox(rows)
        else:
            return self.node.widgets._show()


def row_label(label):
    return Label(value=label, layout=Layout(width='10%'))


def default_layout_widgets(widgets):
    for widget in widgets.values():
        widget.layout = Layout(width='30%')
        widget.style = {'description_width': '40%'}


class GridWidgets(HasWidgets):

    def __init__(self, node):
        HasWidgets.__init__(self, node)

    def _create_widgets(self):
        self._widgets = OrderedDict()
        self.widgets['x_extent'] = BoundedFloatText(description='x [Å]', value=self.node.x_extent, min=1e-16, step=1,
                                                    max=1e12)
        self.widgets['y_extent'] = BoundedFloatText(description='y [Å]', value=self.node.y_extent, min=1e-16, step=1,
                                                    max=1e12)

        self.widgets['x_gpts'] = BoundedIntText(description='x [gpts]', value=self.node.x_gpts, min=1, step=1, max=1e12)
        self.widgets['y_gpts'] = BoundedIntText(description='y [gpts]', value=self.node.y_gpts, min=1, step=1, max=1e12)
        self.widgets['x_sampling'] = BoundedFloatText(description='x [Å / gpt]', value=self.node.x_sampling, min=1e-16,
                                                      step=1, max=1e12)
        self.widgets['y_sampling'] = BoundedFloatText(description='y [Å / gpt]', value=self.node.y_sampling, min=1e-16,
                                                      step=1, max=1e12)

        def callback(message):
            self.widgets['x_extent'].value = self.node.x_extent
            self.widgets['y_extent'].value = self.node.y_extent
            self.widgets['x_gpts'].value = self.node.x_gpts
            self.widgets['y_gpts'].value = self.node.y_gpts
            self.widgets['x_sampling'].value = self.node.x_sampling
            self.widgets['y_sampling'].value = self.node.y_sampling

        self._link_widgets()
        self._add_callback(callback)
        default_layout_widgets(self.widgets)

    def _show(self):
        extent_box = HBox([row_label('Extent'), self.widgets['x_extent'], self.widgets['y_extent']])
        gpts_box = HBox([row_label('Gpts'), self.widgets['x_gpts'], self.widgets['y_gpts']])
        sampling_box = HBox([row_label('Sampling'), self.widgets['x_sampling'], self.widgets['y_sampling']])
        return VBox([extent_box, gpts_box, sampling_box])


class EnergyWidgets(HasWidgets):

    def __init__(self, node):
        HasWidgets.__init__(self, node)

    def _create_widgets(self):
        self._widgets = OrderedDict()

        self.widgets['voltage'] = BoundedFloatText(description='Voltage [eV]', value=self.node.voltage, min=1e-16,
                                                   step=10e3, max=1e12)
        link_widgets(self.node, self.widgets)

        default_layout_widgets(self.widgets)

    def _show(self):
        box = HBox([row_label('Energy'), self.widgets['voltage']])
        return box


class ApertureWidgets(HasWidgets):

    def __init__(self, node):
        HasWidgets.__init__(self, node)

    def _create_widgets(self):
        self._widgets = OrderedDict()
        self._widgets['radius'] = BoundedFloatText(description='Radius [radians]', value=self.node.radius, min=1e-16,
                                                   step=.01, max=1e12)
        self._widgets['rolloff'] = BoundedFloatText(description='Rolloff [radians]', value=self.node.rolloff, min=1e-16,
                                                    step=.01, max=1e12)
        link_widgets(self.node, self.widgets)
        default_layout_widgets(self.widgets)

    def _show(self):
        return HBox([row_label('Aperture'), self.widgets['radius'], self.widgets['rolloff']])


class CTFWidgets(HasWidgets):

    def __init__(self, node):
        HasWidgets.__init__(self, node)

    def _create_widgets(self):
        symbols = self.node.parametrization.symbols
        self._widgets = OrderedDict(
            [(symbol, FloatText(description=symbol, value=getattr(self.node.parametrization, symbol)))
             for symbol in symbols])

        link_widgets(self.node.parametrization, self.widgets)

    def _show(self):
        row_lengths = [3, 4, 5, 6, 7]
        row_widgets = []
        rows = []
        j = 0
        for i, (symbol, widget) in enumerate(self.widgets.items()):
            row_widgets.append(widget)
            widget.layout = Layout(width='13%')
            widget.style = {'description_width': '30%'}
            j += 1

            if j == row_lengths[0]:
                rows.append(HBox(row_widgets))
                row_widgets = []
                row_lengths.pop(0)
                j = 0

        accordion = Accordion([VBox(rows)], selected_index=None)
        accordion.set_title(0, 'Contrast transfer function')
        return accordion


class ProbeWavesWidgets(HasWidgets):

    def __init__(self, node):
        HasWidgets.__init__(self, node)

    def _show(self):
        return None
