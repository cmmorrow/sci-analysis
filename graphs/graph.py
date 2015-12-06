# matplotlib imports
from matplotlib.pyplot import show, subplot, subplot2grid, plot, grid, yticks, \
    xlabel, ylabel, figure, boxplot, hist, legend

# Numpy imports
from numpy import polyfit, polyval

# Scipy imports
from scipy.stats import probplot

# local imports
from ..data.operations import is_vector, is_iterable, is_dict, drop_nan, drop_nan_intersect
from ..data.vector import Vector


class Graph(object):
    """Creates a matplotlib graph as a side effect."""

    nrows = 1
    ncols = 1
    xsize = 5
    ysize = 5

    def __init__(self, data=None, xname="x", yname="y"):

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
        """Prepares and displays the graph as a side effect."""
        pass


class GraphHisto(Graph):
    """Creates a histogram as a side effect."""

    nrows = 3
    ncols = 1

    def __init__(self, data, bins=20, name="Data", color="green", box_plot=True):
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

    nrows = 1
    ncols = 1
    xsize = 3
    ysize = 3

    def __init__(self, xdata, ydata, xname='x Data', yname='y Data', fit=True, pointstyle='k.', linestyle='r-'):
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

    nrows = 1
    ncols = 2
    xsize = 7.5
    ysize = 5

    def __init__(self, vectors, groups=list(), xname='Categories', yname='Values', nqp=True):
        if not any(is_iterable(v) for v in vectors):
            if is_dict(vectors):
                groups = vectors.keys()
                vectors = vectors.values()
            else:
                print("Provided data is not a sequence")
                pass
        self.prob = []
        self.groups = []
        if not is_iterable(groups):
            groups = []
        super(GraphBoxplot, self).__init__(vectors, xname, yname)
        if not groups:
            groups = range(1, len(self.vector) + 1)
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
        """Return a color based on the given index"""
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
        floor = int(num) / len(colors)
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
