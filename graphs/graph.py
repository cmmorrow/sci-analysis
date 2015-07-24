# matplotlib imports
from matplotlib.pyplot import show, subplot, plot, grid, xlabel, ylabel, figure, boxplot, hist

# Numpy imports
from numpy import polyfit, polyval

# local imports
from ..data.operations import is_tuple, is_vector, is_iterable
from ..data.vector import Vector


class Graph(object):

    nrows = 1
    ncols = 1
    xsize = 5
    ysize = 5

    def __init__(self, data=None, xname="x", yname="y"):
        if is_tuple(data):
            self.vector = (Vector(data[0]), Vector(data[1]))
        if not is_vector(data):
            self.vector = Vector(data)
        self.xname = xname
        self.yname = yname

    def draw(self):
        pass


class GraphHisto(Graph):

    nrows = 2
    ncols = 1

    def __init__(self, data, bins=20, name="Data", color="green", box_plot=True):
        super(GraphHisto, self).__init__(data, name, "Probability")
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
            subplot(self.nrows, self.ncols, 1)
            grid(boxplot(self.vector.data, vert=False, showmeans=True), which='major')
            subplot(self.nrows, self.ncols, 2)
        grid(hist(self.vector.data, self.bins, normed=True, color=self.color))
        ylabel(self.yname)
        xlabel(self.xname)
        show()
        pass


class GraphScatter(Graph):

    nrows = 1
    ncols = 1

    def __init__(self, xdata, ydata, xname='x Data', yname='y Data', fit=True, pointstyle='k.', linestyle='r-'):
        super(GraphScatter, self).__init__((xdata, ydata), xname, yname)
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

    def __init__(self, vectors, groups=[], xname='Categories', yname='Values', probplot=True):
        if not is_iterable(vectors):
            pass
        else:
