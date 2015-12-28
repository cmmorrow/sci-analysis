"""sci_analysis module: graph
Classes:
    Graph - The super class all other sci_analysis graphing classes descend from.
    GraphHisto - Draws a histogram.
    GraphScatter - Draws an x-by-y scatter plot.
    GraphBoxplot - Draws box plots of the provided data as well as an optional probability plot.

"""
from __future__ import absolute_import
from __future__ import print_function
# matplotlib imports
from matplotlib.pyplot import show, subplot, subplot2grid, plot, grid, yticks, \
    xlabel, ylabel, figure, boxplot, hist, legend

# Numpy imports
from numpy import polyfit, polyval

# Scipy imports
from scipy.stats import probplot

# local imports
from ..data.vector import Vector
from ..operations.data_operations import is_vector, is_iterable, is_dict, drop_nan, drop_nan_intersect
# from six.moves import range


class Graph(object):
    """The super class all other sci_analysis graphing classes descend from.
    Classes that descend from Graph should implement the draw method at bare minimum.

    Graph members are nrows, ncols, xsize, ysize, vector, xname and yname. The nrows
    member is the number of graphs that will span vertically. The ncols member is
    the number of graphs that will span horizontally. The xsize member is the horizontal
    size of the graph area. The ysize member is the vertical size of the graph area.
    The vector member the data to be plotted. The xname member is the x-axis label.
    The yname member is the y-axis label.

    """

    nrows = 1
    ncols = 1
    xsize = 5
    ysize = 5

    def __init__(self, data=None, xname="x", yname="y"):
        """Converts the data argument to a Vector object and sets it to the Graph
        object's vector member. Sets the xname and yname arguments as the axis
        labels. The default values are "x" and "y".

        :param data: The data to plot
        :param xname: The x-axis label
        :param yname: The y-axis label
        :return: pass
        """

        # If data is a sequence, create a Vector for each argument
        if any(is_iterable(d) for d in data):
            self.vector = []
            for d in data:
                if not is_vector(d):
                    d = Vector(d)
                if d.is_empty():
                    continue
                self.vector.append(d)

        # Wrap data in a Vector if it isn't already a Vector
        elif not is_vector(data):
            self.vector = Vector(data)
        else:
            self.vector = data

        # Set the graph labels
        self.xname = xname
        self.yname = yname

    def draw(self):
        """Prepares and displays the graph based on the set class members."""
        pass


class GraphHisto(Graph):
    """Draws a histogram.

    New class members are bins, color and box_plot. The bins member is the number
    of histogram bins to draw. The color member is the color of the histogram area.
    The box_plot member is a boolean flag for whether to draw the corresponding
    box plot.
    """

    nrows = 3
    ncols = 1

    def __init__(self, data, bins=20, name="Data", color="green", box_plot=True):
        """GraphHisto constructor.

        :param data: The data to be graphed. This arg sets the vector member.
        :param bins: The number of histogram bins to draw. This arg sets the bins member.
        :param name: The optional x-axis label.
        :param color: The optional color of the histogram as a formmated string.
        :param box_plot: Display the optional boxplot.
        :return: pass
        """
        super(GraphHisto, self).__init__(drop_nan(Vector(data)), name, "Probability")
        self.bins = bins
        self.color = color
        self.box_plot = box_plot
        self.draw()

    def draw(self):
        figure(figsize=(self.xsize, self.ysize))
        if len(self.vector) < self.bins:
            self.bins = len(self.vector)
        if self.bins > len(self.vector):
            self.bins = len(self.vector)
        if self.box_plot:
            subplot2grid((self.nrows, self.ncols), (0, 0), rowspan=1)
            grid(boxplot(self.vector.data, vert=False, showmeans=True), which='major', axis='x')
            xlabel("")
            ylabel("")
            yticks([])
            subplot2grid((self.nrows, self.ncols), (1, 0), rowspan=2)
        grid(hist(self.vector.data, self.bins, normed=True, color=self.color))
        ylabel(self.yname)
        xlabel(self.xname)
        show()
        pass


class GraphScatter(Graph):
    """Draws an x-by-y scatter plot.

    Unique class members are fit and style. The fit member is a boolean flag for
    whether to draw the linear best fit line. The style member is a tuple of
    formatted strings that set the matplotlib point style and line style. It is
    also worth noting that the vector member for the GraphScatter class is a
    tuple of xdata and ydata.
    """

    nrows = 1
    ncols = 1
    xsize = 3
    ysize = 3

    def __init__(self, xdata, ydata, xname='x Data', yname='y Data', fit=True, pointstyle='k.', linestyle='r-'):
        """GraphScatter constructor.

        :param xdata: The x-axis data.
        :param ydata: The y-axis data.
        :param xname: The optional x-axis label.
        :param yname: The optional y-axis label.
        :param fit: Display the optional line fit.
        :param pointstyle: The optional matplotlib point style formatted string.
        :param linestyle: The optional matplotlib line style formatted string.
        :return: pass
        """
        super(GraphScatter, self).__init__(drop_nan_intersect(xdata, ydata), xname, yname)
        self.fit = fit
        self.style = (pointstyle, linestyle)
        self.draw()

    def draw(self):
        x = self.vector[0].data
        y = self.vector[1].data
        pointstyle = self.style[0]
        linestyle = self.style[1]
        p = polyfit(x, y, 1, full=True)
        grid(plot(x, y, pointstyle))
        if self.fit:
            plot(x, polyval(p[0], x), linestyle)
        xlabel(self.xname)
        ylabel(self.yname)
        show()
        pass


class GraphBoxplot(Graph):
    """Draws box plots of the provided data as well as an optional probability plot.

    Unique class members are groups, nqp and prob. The groups member is a list of
    labels for each boxplot. If groups is an empty list, sequentially ascending
    numbers are used for each boxplot. The nqp member is a flag that turns the
    probability plot on or off. The prob member is a list of tuples that contains
    the data used to graph the probability plot. It is also worth noting that the
    vector member for the GraphBoxplot is a list of lists that contain the data
    for each boxplot.
    """

    nrows = 1
    ncols = 2
    xsize = 7.5
    ysize = 5

    def __init__(self, vectors, groups=list(), xname='Categories', yname='Values', nqp=True):
        """GraphBoxplot constructor. NOTE: If vectors is a dict, the boxplots are
        graphed in random order instead of the provided order.

        :param vectors: A list of lists or dict of lists of the data to graph.
        :param groups: An optional list of boxplot labels. The order should match the order in vectors.
        :param xname: The optional x-axis label for the boxplots.
        :param yname: The optional y-axis label.
        :param nqp: Display the optional probability plot.
        :return: pass
        """
        if not any(is_iterable(v) for v in vectors):
            if is_dict(vectors):
                groups = list(vectors.keys())
                vectors = list(vectors.values())
            else:
                print("Provided data is not a sequence")
                pass
        self.prob = []
        self.groups = []
        if not is_iterable(groups):
            groups = []
        super(GraphBoxplot, self).__init__(vectors, xname, yname)
        if not groups:
            groups = list(range(1, len(self.vector) + 1))
        for i, v in enumerate(self.vector):
            self.vector[i] = v = drop_nan(v)
            if len(v) == 0:
                groups = groups[:i] + groups[i + 1:]
                continue
            if nqp:
                q, fit = probplot(v)
                self.prob.append((q, fit))
        self.groups = groups
        self.draw()

    def get_color(self, num):
        """Return a color based on the given num argument.

        :param num: An integer not equal to zero that returns a corresponding color
        :return: A color tuple calculated from the num argument
        """
        colors = [(0,   0,   1, 1),
                  (0,   0.5, 0, 1),
                  (1,   0,   0, 1),
                  (0,   1,   1, 1),
                  (1,   1,   0, 1),
                  (1,   0,   1, 1),
                  (1,   0.5, 0, 1),
                  (0.5, 0,   1, 1),
                  (0.5, 1,   0, 1),
                  (1,   1,   1, 1)
                  ]
        desired_color = []
        if num < 0:
            num *= -1
        floor = int(num) // len(colors)
        remainder = int(num) % len(colors)
        selected = colors[remainder]
        if floor > 0:
            for value in selected:
                desired_color.append(value / (2.0 * floor) + 0.4)
            return tuple(desired_color)
        else:
            return selected

    def draw(self):
        if len(self.prob) > 0:
            figure(figsize=(self.xsize * 2, self.ysize))
            ax2 = subplot(self.nrows, self.ncols, 2)
            grid(which='major')
            for i, g in enumerate(self.prob):
                plot(g[0][0], g[0][1], marker='^', color=self.get_color(i), label=self.groups[i])
                plot(g[0][0], g[1][0] * g[0][0] + g[1][1], linestyle='-', color=self.get_color(i))
            legend(loc='best')
            xlabel("Quantiles")
            subplot(self.nrows, self.ncols, 1, sharey=ax2)
        else:
            figure(figsize=(self.xsize, self.ysize))
        grid(boxplot(self.vector, showmeans=True, labels=self.groups))
        ylabel(self.yname)
        xlabel(self.xname)
        show()
        pass
