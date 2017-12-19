"""sci_analysis module: graph
Classes:
    Graph - The super class all other sci_analysis graphing classes descend from.
    GraphHisto - Draws a histogram.
    GraphScatter - Draws an x-by-y scatter plot.
    GraphBoxplot - Draws box plots of the provided data as well as an optional probability plot.

"""
# TODO: Add preferences back in a future version
# from ..preferences.preferences import GraphPreferences
# from six.moves import range


_colors = (
    (0.0, 0.3, 0.7),    # blue
    (1.0, 0.1, 0.1),    # red
    (0.0, 0.7, 0.3),    # green
    (1.0, 0.5, 0.0),    # orange
    (0.1, 1.0, 1.0),    # cyan
    (1.0, 1.0, 0.0),    # yellow
    (1.0, 0.0, 1.0),    # magenta
    (0.5, 0.0, 1.0),    # purple
    (0.5, 1.0, 0.0),    # light green
    (0.0, 0.0, 0.0)     # black
)

_color_names = (
    'blue',
    'red',
    'green',
    'orange',
    'cyan',
    'yellow',
    'magenta',
    'purple',
    'light green',
    'black'
)


class Graph(object):
    """The super class all other sci_analysis graphing classes descend from.
    Classes that descend from Graph should implement the draw method at bare minimum.

    Graph members are _nrows, _ncols, _xsize, _ysize, _data, _xname and _yname. The _nrows
    member is the number of graphs that will span vertically. The _ncols member is
    the number of graphs that will span horizontally. The _xsize member is the horizontal
    size of the graph area. The _ysize member is the vertical size of the graph area.
    The _data member the data to be plotted. The _xname member is the x-axis label.
    The _yname member is the y-axis label.

    Parameters
    ----------
    _nrows : int, static
        The number of graphs that will span vertically.
    _ncols : int, static
        The number of graphs that will span horizontally.
    _xsize : int, static
        The horizontal size of the graph area.
    _ysize : int, static
        The vertical size of the graph area.
    _min_size : int, static
        The minimum required length of the data to be graphed.
    _xname : str
        The x-axis label.
    _yname : str
        The y-axis label.
    _data : Data or list(d1, d2, ..., dn)
        The data to graph.

    Returns
    -------
    pass
    """

    _nrows = 1
    _ncols = 1
    _xsize = 5
    _ysize = 5
    _min_size = 1

    def __init__(self, data, **kwargs):

        self._xname = kwargs['xname'] if 'xname' in kwargs else 'x'
        self._yname = kwargs['yname'] if 'yname' in kwargs else 'y'
        self._data = data

    def get_color_by_name(self, color='black'):
        """Return a color array based on the string color passed.

        Parameters
        ----------
        color : str
            A string color name.

        Returns
        -------
        color : tuple
            A color tuple that corresponds to the passed color string.

        """
        return self.get_color(_color_names.index(color))

    @staticmethod
    def get_color(num):
        """Return a color based on the given num argument.

        Parameters
        ----------
        num : int
            A numeric value greater than zero that returns a corresponding color.

        Returns
        -------
        color : tuple
            A color tuple calculated from the num argument.
        """
        desired_color = []
        floor = int(num) // len(_colors)
        remainder = int(num) % len(_colors)
        selected = _colors[remainder]
        if floor > 0:
            for value in selected:
                desired_color.append(value / (2.0 * floor) + 0.4)
            return tuple(desired_color)
        else:
            return selected

    def draw(self):
        """
        Prepares and displays the graph based on the set class members.
        """
        raise NotImplementedError
