import traitlets
from ipywidgets import Layout, GridBox


def link_widgets(o, widgets):
    for name, widget in widgets.items():
        if hasattr(o, name):
            traitlets.link((o, name), (widget, 'value'))


def layout_widgets(widgets, n=4, width=100):
    grid_template_columns = ' '.join(['{}%'.format(100 / n)] * n)

    rows = []
    for i, (name, widget) in enumerate(widgets.items()):
        if i % n == 0:
            rows.append([])

        widget.layout = Layout(width='auto', grid_area=name)
        rows[-1].append(name)

    for i, row in enumerate(rows):
        rows[i] = ' '.join(row + ['.'] * (n - len(row)))

    placements = '"' + '" \n"'.join(rows) + '"'

    layout = Layout(width='{}%'.format(width), grid_template_rows='auto',
                    grid_template_columns=grid_template_columns,
                    grid_template_areas='''{}'''.format(placements))

    return GridBox(children=list(widgets.values()), layout=layout)
