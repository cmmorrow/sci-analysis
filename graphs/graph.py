# matplotlib imports
from matplotlib.pyplot import show, subplot, grid, xlabel, ylabel, figure, boxplot, hist

# local imports
from ..data import operations, vector


class Graph(object):

    nrows = 1
    ncols = 1
    xsize = 5
    ysize = 5

    def __init__(self, data=None, xname="x", yname="y"):
        if not operations.is_vector(data):
            self.data = vector.Vector(data)
        self.xname = xname
        self.yname = yname
        self.draw()

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

    def draw(self):
        figure(figsize=(5, 5))
        if len(self.data) < self.bins:
            self.bins = len(self.data)
        if self.bins > len(self.data):
            self.bins = len(self.data)
        if self.box_plot:
            subplot(self.nrows, self.ncols, 2)
            grid(boxplot(self.data.data, vert=False, showmeans=True), which='major')
            subplot(self.nrows, self.ncols, 1)
        grid(hist(self.data.data, normed=True, color=self.color))
        ylabel(self.yname)
        xlabel(self.xname)
        show()
        pass
