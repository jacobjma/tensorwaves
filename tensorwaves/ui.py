import traitlets
import ipywidgets as widgets

description_layout = widgets.Layout(width='240px')
widget_layout = widgets.Layout(width='240px')



class InterfaceBuilder(object):

    def __init__(self, instance, dig=True):
        self.instance = instance
        graph = calculation_graph(instance)
        graph = flatten_hierarchical_graph(graph)
        if dig is False:
            graph = {instance: graph[instance]}

        self.widgets = {}
        for node, traits in graph.items():
            self.widgets[node] = []
            for trait in traits:
                try:
                    widget_instruct = getattr(node, '_{}_widget'.format(trait.name))
                    kwargs = {'style': {'description_width': '80px'},
                              'layout': widgets.Layout(width='240px')}
                    kwargs.update(widget_instruct['kwargs'])
                    self.widgets[node].append(widget_instruct['widget'](**kwargs))
                    traitlets.link((node, trait.name), (self.widgets[node][-1], 'value'))
                except AttributeError:
                    pass

    def autogenerate(self):
        widget_boxes = []
        labels = []
        for key, value in self.widgets.items():
            if len(value) > 0:
                labels.append(widgets.Label(key.description))
                widget_boxes.append(widgets.HBox(value))
        return widgets.HBox([widgets.VBox(labels), widgets.VBox(widget_boxes)])

    def link(self, callable):
        for key, widget_list in self.widgets.items():
            for widget in widget_list:
                widget.observe(callable, 'value')

    def unlink(self, callable):
        for key, widget_list in self.widgets.items():
            for widget in widget_list:
                widget.unobserve(callable, 'value')
