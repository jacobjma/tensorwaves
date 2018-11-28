def tanh_sinh_quadrature(m, h):
    xk = tf.Variable(tf.zeros(2 * m))
    wk = tf.Variable(tf.zeros(2 * m))
    for i in range(0, 2 * m):
        k = i - m
        xk[i].assign(tf.tanh(math.pi / 2 * tf.sinh(k * h)))
        numerator = h / 2 * math.pi * tf.cosh(k * h)
        denominator = tf.cosh(math.pi / 2 * tf.sinh(k * h)) ** 2
        wk[i].assign(numerator / denominator)
    return xk, wk


descriptions = ('\(C_{1,0}\)', '\(C_{1,2}\)', '\(\phi_{1,2}\)',
                '\(C_{2,1}\)', '\(\phi_{2,1}\)', '\(C_{2,3}\)', '\(\phi_{2,3}\)',
                '\(C_{3,0}\)', '\(C_{3,2}\)', '\(\phi_{3,2}\)', '\(C_{3,4}\)', '\(\phi_{3,4}\)',
                '\(C_{4,1}\)', '\(\phi_{4,1}\)', '\(C_{4,3}\)', '\(\phi_{4,3}\)', '\(C_{4,5}\)', '\(\phi_{4,5}\)',
                '\(C_{5,0}\)', '\(C_{5,2}\)', '\(\phi_{5,2}\)', '\(C_{5,4}\)', '\(\phi_{5,4}\)', '\(C_{5,6}\)',
                '\(\phi_{5,6}\)')


def interface(self):
    widgets = OrderedDict(
        [(symbol, ipywidgets.FloatText(description=description, value=getattr(self, symbol),
                                       style={'description_width': '30px'})) for
         symbol, description in zip(self.symbols, self.descriptions)])

    self._link_widgets(widgets)

    n_columns = 7
    grid_template_columns = ' '.join(['{}%'.format(100 / n_columns)] * n_columns)

    new_line = [3, 7, 12, 18, 25]
    rows = [[]]
    for i, (name, widget) in enumerate(widgets.items()):
        if i == new_line[0]:
            rows[-1] = ' '.join(rows[-1] + ['.'] * (n_columns - len(rows[-1])))
            rows.append([])
            new_line.pop(0)

        widget.layout = ipywidgets.Layout(width='auto', grid_area=name)
        rows[-1].append(name)

    rows[-1] = ' '.join(rows[-1] + ['.'] * (n_columns - len(rows[-1])))

    placements = '"' + '" \n"'.join(rows) + '"'

    layout = ipywidgets.Layout(width='100%', grid_template_rows='auto',
                               grid_template_columns=grid_template_columns,
                               grid_template_areas='''{}'''.format(placements))

    box = ipywidgets.GridBox(children=list(widgets.values()), layout=layout)

    accordion = ipywidgets.Accordion(children=[box])
    accordion.set_title(0, 'All aberrations')
    return accordion

def realspace_progagator(grid, wavelength):
    """ Real space fresnel propagator """
    x, y = grid.x_axis.grid(), grid.y_axis.grid()
    r = np.pi / (wavelength * grid.hz) * ((x ** 2)[:, None] + (y ** 2)[None, :])
    return tf.complex(tf.sin(r), - tf.cos(r)) / (wavelength * grid.hz)