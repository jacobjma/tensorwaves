import numbers
from math import ceil
from collections import OrderedDict

import ipywidgets
import traitlets

from tensorwaves import utils

DEFAULT_SAMPLING = .05
MAX_GPTS = 1e1 / utils.EPS


def _flatten_graph(graph):
    flattened = {}
    top = next(iter(graph.keys()))
    queue = [(top, graph[top])]
    while queue:
        name, graph = queue.pop()
        flattened[name] = []
        for key, value in graph.items():
            if isinstance(value, dict):
                queue.append((key, value))
            else:
                flattened[name].append(value)

    return flattened


def _dig_upstream(node, graph):
    for name, trait in node.traits().items():
        if isinstance(trait, traitlets.Instance):
            if (Node in trait.klass.__bases__) | (TensorFactory in trait.klass.__bases__):
                new_node = trait.get(node)
                graph[new_node] = {}
                _dig_upstream(new_node, graph[new_node])
        else:
            graph[name] = trait

    return graph


class Node(traitlets.HasTraits):
    _widgets = {}

    def _build_widgets(self):
        return {}

    def _link_widgets(self, widgets):
        for name, widget in widgets.items():
            traitlets.link((self, name), (widget, 'value'))

    def _layout_widgets(self, widgets):
        return ipywidgets.HBox(list(widgets.values()))

    def _interface(self):
        self._widgets = self._build_widgets()
        self._link_widgets(self._widgets)
        return self._layout_widgets(self._widgets)

    def get_widgets(self, include_upstream=True):
        if include_upstream:
            graph = _flatten_graph(self.upstream_traits())
            widgets = {}
            for node in graph.keys():
                widgets[node] = node._widgets
            return widgets
        else:
            return self._widgets

    def interface(self, include_upstream=True):
        if include_upstream:
            graph = _flatten_graph(self.upstream_traits())
            box = []
            for node in graph.keys():
                box.append(node._interface())
            return ipywidgets.VBox(box)
        else:
            return self._interface()

    def graph_observe(self, callable):
        for key, widget_list in self._widgets.items():
            for widget in widget_list:
                widget.observe(callable, 'value')

    def graph_unobserve(self, callable):
        for key, widget_list in self._widgets.items():
            for widget in widget_list:
                widget.unobserve(callable, 'value')

    def upstream_traits(self):
        graph = {}
        graph[self] = {}
        graph[self] = _dig_upstream(self, graph[self])
        return graph


class Grid(Node):
    description = traitlets.Unicode(default_value='Grid')

    length = traitlets.Float(default_value=10, min=utils.EPS)
    gpts = traitlets.Int(default_value=200, min=1, max=MAX_GPTS)
    sampling = traitlets.Float(default_value=.05, min=utils.EPS)
    adjusted_sampling = traitlets.Float(default_value=.05, min=utils.EPS)

    def _gpts2sampling(self, sampling):
        return int(ceil(self.length / sampling))

    def _sampling2gpts(self, gpts):
        return self.length / gpts

    @traitlets.observe('length', 'gpts', 'sampling')
    def _observe_all(self, _):
        self.adjusted_sampling = self.length / self._gpts2sampling(self.sampling)

    @traitlets.observe('length')
    def _observe_length(self, _):
        old_sampling = self.sampling
        self.gpts = self._gpts2sampling(self.sampling)
        self.sampling = old_sampling

    @traitlets.observe('gpts')
    def _observe_gpts(self, change):
        self.sampling = self._sampling2gpts(change['new'])

    @traitlets.observe('sampling')
    def _observe_sampling(self, change):
        self.gpts = self._gpts2sampling(change['new'])

    def _build_widgets(self):
        widgets = OrderedDict()
        widgets['length'] = ipywidgets.BoundedFloatText(description='Length', value=self.length, min=utils.EPS)
        widgets['gpts'] = ipywidgets.BoundedIntText(description='Grid points', value=self.gpts, min=1, max=MAX_GPTS)
        widgets['sampling'] = ipywidgets.BoundedFloatText(description='Sampling', value=self.sampling, min=utils.EPS)
        return widgets


class Accelerator(Node):
    description = traitlets.Unicode(default_value='Accelerator')

    energy = traitlets.Float(default_value=100e3)
    wavelength = traitlets.Float()

    @traitlets.default('wavelength')
    def _observe_energy(self):
        return utils.energy2wavelength(self.energy)

    @traitlets.observe('energy')
    def _observe_energy(self, change):
        self.wavelength = utils.energy2wavelength(change['new'])

    def get_interaction_parameter(self):
        if self.energy is None:
            return None
        else:
            return utils.energy2sigma(self.energy)

    def _build_widgets(self):
        widgets = OrderedDict()
        widgets['energy'] = ipywidgets.BoundedFloatText(description='Energy', value=self.energy, min=utils.EPS,
                                                        step=10e3, max=1e10)
        return widgets


class Space(Node):
    name = traitlets.Unicode(default_value='space')
    description = traitlets.Unicode(default_value='Space')

    space = traitlets.Unicode(default_value='direct')
    _space_summary = lambda x: 'Calculated in {} space'.format(x)

    @traitlets.validate('space')
    def _validate_space(self, proposal):
        if proposal['value'].lower() not in ('direct', 'fourier', 'hybrid'):
            raise traitlets.TraitError()
        else:
            return proposal['value']


def parse_xy_grid(extent=None, gpts=None, sampling=None):
    x_grid = Grid(description='X Grid')
    y_grid = Grid(description='Y Grid')

    if extent is not None:
        x_grid.extent = extent if isinstance(extent, numbers.Number) else extent[0]
        y_grid.extent = extent if isinstance(extent, numbers.Number) else extent[1]

    if gpts is not None:
        x_grid.gpts = gpts if isinstance(gpts, numbers.Number) else gpts[0]
        y_grid.gpts = gpts if isinstance(gpts, numbers.Number) else gpts[1]

    if sampling is not None:
        x_grid.extent = sampling if isinstance(sampling, numbers.Number) else sampling[0]
        y_grid.extent = sampling if isinstance(sampling, numbers.Number) else sampling[1]

    return x_grid, y_grid


def parse_accelerator(energy):
    accelerator = Accelerator()
    if energy is not None:
        accelerator.energy = energy
    return accelerator


def parse_space(space):
    accelerator = Space()
    if space is not None:
        accelerator.space = space
    return accelerator


def trait2property(name, has_traits):
    """Helper function to easily create attribute property from trait."""

    def getter(self):
        return self.get(name, has_traits)

    def setter(self, value):
        self.set(name, has_traits, value)

    return property(getter, setter)


class TensorFactory(Node):
    x_grid = traitlets.Instance(Grid)
    y_grid = traitlets.Instance(Grid)

    def __init__(self, extent=None, gpts=None, sampling=None):
        self.x_grid, self.y_grid = parse_xy_grid(extent, gpts, sampling)
        super().__init__()

    gpts = trait2property('gpts', ('x_grid', 'y_grid'))
    sampling = trait2property('sampling', ('x_grid', 'y_grid'))
    extent = trait2property('length', ('x_grid', 'y_grid'))

    def get(self, name, has_traits):
        return (getattr(getattr(self, has_traits[0]), name),
                getattr(getattr(self, has_traits[1]), name))

    def set(self, name, has_traits, value):
        if isinstance(value, numbers.Number) | (value is None):
            value = (value, value)
        setattr(getattr(self, has_traits[0]), name, value[0])
        setattr(getattr(self, has_traits[1]), name, value[1])
