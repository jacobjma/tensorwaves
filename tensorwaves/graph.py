class Node(object):
    pass


def flatten_graph(graph):
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


def _dig_upstream(has_traits, graph):

    for name, trait in has_traits.traits().items():

        trait_content = trait.get(has_traits)

        if Node in trait_content.__class__.__bases__:
            trait_content.traits()
            graph[trait_content] = {}
            _dig_upstream(trait_content, graph[trait_content])
        else:
            graph[name] = trait

    return graph


def upstream_traits(has_traits):
    graph = {}
    graph[has_traits] = {}
    graph[has_traits] = _dig_upstream(has_traits, graph[has_traits])
    return graph
