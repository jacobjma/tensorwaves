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


class Node(object):

    def __init__(self, incoming=None):
        if incoming is None:
            self._incoming = {}
        else:
            self._incoming = incoming

    @property
    def incoming(self):
        return self._incoming

    def _dig_upstream(self, node, graph):
        for name, incoming_node in node.incoming.items():
            graph[incoming_node] = {}
            self._dig_upstream(incoming_node, graph[incoming_node])

        return graph

    def upstream(self, flatten=False):
        graph = {}
        graph[self] = {}
        graph[self] = self._dig_upstream(self, graph[self])
        if flatten:
            return flatten_graph(self.upstream())
        else:
            return graph

    def upstream_nodes(self):
        return list(self.upstream(flatten=True).keys())

class SymmetricCTF(ParameterizedCTF):

    def __init__(self):
        self.symbols = ('C10', 'C30', 'C50')
        self.aliases = ('defocus', 'Cs', 'C5')

        ParameterizedCTF.__init__(self, self.symbols, self.aliases)

    def get_function(self):
        chi = None
        if self.C10 != 0.:
            chi = lambda a, a2: 1 / 2. * a2 * self.C10
        if self.C30 != 0.:
            chi_old1 = chi
            chi = lambda a, a2: chi_old1(a, a2) + 1 / 4. * a2 ** 2 * self.C30
        if self.C50 != 0.:
            chi_old2 = chi
            chi = lambda a, a2: chi_old2(a, a2) + 1 / 6. * a2 ** 3 * self.C50

        if chi is None:
            return lambda a, a2, b: tf.ones(a.shape)
        else:
            return chi

    def copy(self):
        c = self.__class__()
        for symbol in self.symbols:
            setattr(c, symbol, getattr(self, symbol))
        return c


class CartesianCTF(ParameterizedCTF):

    def __init__(self):
        self.symbols = ('C10', 'C12a', 'C12b',
                        'C21a', 'C21b', 'C23a', 'C23b',
                        'C30', 'C32a', 'C32b', 'C34a', 'C34b')

        self.aliases = ('defocus', 'astig_x', 'astig_y',
                        'coma_x', 'coma_y', 'astig_x_2', 'astig_y_2',
                        'Cs', None, None, None, None)

        ParameterizedCTF.__init__(self, self.symbols, self.aliases)

    def get_function(self):
        chi = None
        # todo: implement 4th and 5th order
        if any([getattr(self, symbol) != 0. for symbol in ('C10', 'C12a', 'C12b')]):
            chi = lambda ax, ay, ax2, ay2, a2: (1 / 2. * (self.C10 * a2 +
                                                          self.C12a * (ax2 - ay2)) + self.C12b * ax * ay)
        if any([getattr(self, symbol) != 0. for symbol in ('C21a', 'C21b', 'C23a', 'C23b')]):
            chi_old1 = chi
            chi = (lambda ax, ay, ax2, ay2, a2: chi_old1(ax, ay, ax2, ay2, a2) +
                                                1 / 3. * (a2 * (self.C21a * ax + self.C21b * ay) +
                                                          self.C23a * ax * (ax2 - 3 * ay2) +
                                                          self.C23b * ay * (ay2 - 3 * ax2)))
        if any([getattr(self, symbol) != 0. for symbol in ('C30', 'C32a', 'C32b', 'C34a', 'C34b')]):
            chi_old2 = chi
            chi = (lambda ax, ay, ax2, ay2, a2: chi_old2(ax, ay, ax2, ay2, a2) +
                                                1 / 4. * (self.C30 * a2 ** 2 +
                                                          self.C32a * (ax2 ** 2 - ay2 ** 2) +
                                                          2 * self.C32b * ax * ay * a2 +
                                                          self.C34a * (ax2 ** 2 - 6 * ax2 * ay2 + ay2 ** 2) +
                                                          4 * self.C34b * (ax * ay2 * ay - ax2 * ax * ay)))
        if chi is None:
            return lambda ax, ay, ax2, ay2, a2: tf.ones(ax.shape)
        else:
            return chi

    def copy(self):
        c = self.__class__()
        for symbol in self.symbols:
            setattr(c, symbol, getattr(self, symbol))
        return c

def polar2cartesian(polar):
    cartesian = {}
    cartesian['C10'] = polar['C10']
    cartesian['C12a'] = - polar['C12'] * tf.cos(2 * polar['phi12'])
    cartesian['C12b'] = polar['C12'] * tf.cos(pi / 2 - 2 * polar['phi12'])
    cartesian['C21a'] = polar['C21'] * tf.cos(pi / 2 - polar['phi21'])
    cartesian['C21b'] = polar['C21'] * tf.cos(polar['phi21'])
    cartesian['C23a'] = polar['C23'] * tf.cos(3 * pi / 2. - 3 * polar['phi23'])
    cartesian['C23b'] = polar['C23'] * tf.cos(3 * polar['phi23'])
    cartesian['C30'] = polar['C30']
    cartesian['C32a'] = - polar['C32'] * tf.cos(2 * polar['phi32'])
    cartesian['C32b'] = polar['C32'] * tf.cos(pi / 2 - 2 * polar['phi32'])
    cartesian['C34a'] = polar['C34'] * tf.cos(-4 * polar['phi34'])
    K = tf.sqrt(3 + tf.sqrt(8.))
    cartesian['C34b'] = 1 / 4. * (1 + K ** 2) ** 2 / (K ** 3 - K) * polar['C34'] * tf.cos(
        4 * tf.atan(1 / K) - 4 * polar['phi34'])
    return cartesian



def show_line(self, phi=0, k_max=1):
    k = tf.linspace(0., k_max, 1024)
    alpha = self.wavelength * k

    chi = 2 * pi / self.wavelength * self.parametrization.get_function()(alpha, alpha ** 2, phi)

    H = complex_exponential(-chi)

    if np.isfinite(self.aperture_radius):
        aperture = Aperture(radius=self.aperture_radius, rolloff=self.aperture_rolloff)._function(alpha)
        H *= tf.cast(aperture, tf.complex64)
        show_line(k, aperture, mode='real', label='aperture')

    if self.focal_spread > 0.:
        temporal = TemporalEnvelope(focal_spread=self.focal_spread)._function(alpha)
        H *= tf.cast(temporal, tf.complex64)
        show_line(k, temporal, mode='real', label='temporal')

    show_line(k, H, mode='real', label='real')
    show_line(k, H, mode='imaginary', label='imaginary')
    plt.legend()