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
    xlabel, ylabel, figure, boxplot, hist, legend, setp, savefig, contour
from matplotlib.gridspec import GridSpec

# Numpy imports
from numpy import polyfit, polyval, sort, arange, array, linspace, mgrid, vstack, reshape

# Scipy imports
from scipy.stats import probplot, gaussian_kde

# local imports
from ..data.vector import Vector
from ..operations.data_operations import is_vector, is_iterable, is_dict, drop_nan, drop_nan_intersect
# TODO: Add preferences back in a future version
# from ..preferences.preferences import GraphPreferences
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

    def __init__(self, data=None, xname="x", yname="y", save_to=None):
        """Converts the data argument to a Vector object and sets it to the Graph
        object's vector member. Sets the xname and yname arguments as the axis
        labels. The default values are "x" and "y".

        :param data: The data to plot
        :param xname: The x-axis label
        :param yname: The y-axis label
        :param save_to: Save the graph to the specified path
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

        # Set the save path
        self.file = save_to

    def draw(self):
        """Prepares and displays the graph based on the set class members."""
        pass

    #TODO: Finish implementing save_graph and remove the sub class method
    def save_graph(self):
        if self.file:
            savefig(self.file)



class GraphHisto(Graph):
    """Draws a histogram.

    New class members are bins, color and box_plot. The bins member is the number
    of histogram bins to draw. The color member is the color of the histogram area.
    The box_plot member is a boolean flag for whether to draw the corresponding
    box plot.
    """

    #nrows = 2
    #ncols = 1
    ysize = 4

    def __init__(self, data,
                 bins=20,
                 name="Data",
                 distribution='norm',
                 color='green',
                 box_plot=True,
                 cdf=False,
                 violin_plot=False,
                 histogram=True,
                 fit=False,
                 save_to=None):
        """GraphHisto constructor.

        :param data: The data to be graphed. This arg sets the vector member.
        :param bins: The number of histogram bins to draw. This arg sets the bins member.
        :param name: The optional x-axis label.
        :param distribution: The theoretical distribution to fit.
        :param color: The optional color of the histogram as a formmated string.
        :param box_plot: Toggle the display of the optional boxplot.
        :param cdf: Toggle the display of the optional cumulative density function plot.
        :param violin_plot: Add a distribution density overlay to the boxplot.
        :param fit: Toggle the display of the best fit line for the specified distribution.
        :param save_to: Save the graph to the specified path
        :return: pass
        """
        super(GraphHisto, self).__init__(drop_nan(Vector(data)), name, "Probability", save_to)
        self.bins = bins
        self.distribution = distribution
        self.color = color
        # if GraphPreferences.Plot.boxplot != GraphPreferences.Plot.defaults[0]:
        #    _box_plot = GraphPreferences.Plot.boxplot
        # else:
        #    _box_plot = box_plot
        self.box_plot = box_plot
        self.violin_plot = violin_plot
        self.histogram = histogram
        # if GraphPreferences.Plot.cdf != GraphPreferences.Plot.defaults[2]:
        #    _cdf = GraphPreferences.Plot.cdf
        # else:
        #    _cdf = cdf
        self.cdf = cdf
        self.fit = fit
        self.draw()

    def fit_distro(self):
        if not self.distribution:
            self.distribution = 'norm'
        distro_class = getattr(__import__('scipy.stats', globals(), locals(), [self.distribution], -1), self.distribution)
        parms = distro_class.fit(self.vector)
        distro = linspace(distro_class.ppf(0.001, *parms), distro_class.ppf(0.999, *parms), 100)
        distro_pdf = distro_class.pdf(distro, *parms)
        if self.fit:
            distro_cdf = distro_class.cdf(distro, *parms)
        else:
            distro_cdf = None
        return distro, distro_pdf, distro_cdf

    def calc_cdf(self):
        x_sorted_vector = sort(self.vector)
        if len(x_sorted_vector) == 0:
            return 0, 0
        y_sorted_vector = arange(len(x_sorted_vector) + 1) / float(len(x_sorted_vector))
        x_cdf = array([x_sorted_vector, x_sorted_vector]).T.flatten()
        y_cdf = array([y_sorted_vector[:(len(y_sorted_vector)-1)], y_sorted_vector[1:]]).T.flatten()
        return x_cdf, y_cdf

    def draw(self):
        histo_span = 3
        box_plot_span = 1
        cdf_span = 3
        h_ratios = [histo_span]
        p = []
        if self.box_plot:
            self.ysize += 1
            self.nrows += 1
            h_ratios.insert(0, box_plot_span)
        if self.cdf:
            self.ysize += 4
            self.nrows += 1
            h_ratios.insert(0, cdf_span)
        f = figure(figsize=(self.xsize, self.ysize))
        gs = GridSpec(self.nrows, self.ncols, height_ratios=h_ratios, hspace=0)
        if len(self.vector) < self.bins:
            self.bins = len(self.vector)
        if self.fit:
            distro, distro_pdf, distro_cdf = self.fit_distro()
        if self.cdf:
            x_cdf, y_cdf = self.calc_cdf()
            ax_cdf = f.add_subplot(gs[0])
            grid(ax_cdf.plot(x_cdf, y_cdf, 'k-'))
            p.append(ax_cdf.get_xticklabels())
            if self.fit:
                ax_cdf.plot(distro, distro_cdf, 'r--', linewidth=2)
            yticks(arange(11) * 0.1)
            ylabel("Cumulative Probability")
        if self.box_plot or self.violin_plot:
            if self.cdf:
                ax_box = f.add_subplot(gs[len(h_ratios)-2], sharex=ax_cdf)
            else:
                ax_box = f.add_subplot(gs[len(h_ratios)-2])
            # if GraphPreferences.distribution['violin']:
            if self.violin_plot:
                grid(ax_box.violinplot(self.vector.data, vert=False, showextrema=False, showmedians=False, showmeans=False))
            # if GraphPreferences.distribution['boxplot']:
            grid(ax_box.boxplot(self.vector.data, vert=False, showmeans=True))
            yticks([])
            p.append(ax_box.get_xticklabels())
            ax_hist = f.add_subplot(gs[len(h_ratios)-1], sharex=ax_box)
        else:
            ax_hist = f.add_subplot(gs[len(h_ratios)-1])
        grid(ax_hist.hist(self.vector.data, self.bins, normed=True, color=self.color))
        if self.fit:
            ax_hist.plot(distro, distro_pdf, 'r--', linewidth=2)
        if len(p) > 0:
            setp(p, visible=False)
        ylabel(self.yname)
        xlabel(self.xname)
        show()
        if self.file:
            savefig(self.file)
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
    xsize = 5
    ysize = 5

    def __init__(self, xdata,
                 ydata,
                 xname='x Data',
                 yname='y Data',
                 fit=True,
                 pointstyle='k.',
                 linestyle='r-',
                 points=True,
                 contours=False,
                 num_of_contours=31,
                 contour_width=1.1,
                 histogram_borders=False,
                 bins=20,
                 color='green',
                 boxplot_borders=False,
                 violin_plots=True,
                 save_to=None):
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
        super(GraphScatter, self).__init__(drop_nan_intersect(Vector(xdata), Vector(ydata)), xname, yname, save_to)
        self.fit = fit
        self.style = (pointstyle, linestyle)
        self.points = points
        self.contours = contours
        self.contour_props = (num_of_contours, contour_width)
        self.histogram_props = (bins, color)
        self.histogram_borders = histogram_borders
        self.boxplot_borders = boxplot_borders
        self.violin_plots = violin_plots
        self.draw()

    def calc_contours(self):
        xmin = self.vector[0].data.min()
        xmax = self.vector[0].data.max()
        ymin = self.vector[1].data.min()
        ymax = self.vector[1].data.max()

        values = vstack([self.vector[0].data, self.vector[1].data])
        kernel = gaussian_kde(values)
        _x, _y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = vstack([_x.ravel(), _y.ravel()])
        _z = reshape(kernel(positions).T, _x.shape)
        return _x, _y, _z, arange(_z.min(), _z.max(), (_z.max() - _z.min()) / self.contour_props[0])

    def calc_fit(self):
        x = self.vector[0].data
        y = self.vector[1].data
        p = polyfit(x, y, 1, full=True)
        return polyval(p[0], x)

    def draw(self):
        x = self.vector[0].data
        y = self.vector[1].data
        pointstyle = self.style[0]
        linestyle = self.style[1]
        h_ratio = [1, 1]
        w_ratio = [1, 1]
        borders = self.histogram_borders or self.boxplot_borders

        if borders:
            self.nrows, self.ncols = 2, 2
            self.xsize, self.ysize = 7, 6
            h_ratio, w_ratio = [2, 5], [5, 2]
            main_plot = 2
        else:
            main_plot = 0
        f = figure(figsize=(self.xsize, self.ysize))
        if borders:
            gs = GridSpec(self.nrows, self.ncols, height_ratios=h_ratio, width_ratios=w_ratio, hspace=0.1, wspace=0.1)
        else:
            gs = GridSpec(self.nrows, self.ncols)
        ax2 = f.add_subplot(gs[main_plot])
        if self.points:
            grid(ax2.plot(x, y, pointstyle, zorder=1))
        xlabel(self.xname)
        ylabel(self.yname)
        if self.contours:
            x_prime, y_prime, z, levels = self.calc_contours()
            ax2.contour(x_prime, y_prime, z, levels, linewidths=self.contour_props[1], nchunk=16, extend='both', zorder=2)
        if self.fit:
            ax2.plot(x, self.calc_fit(), linestyle, zorder=3)
        if borders:
            ax1 = f.add_subplot(gs[0], sharex=ax2)
            ax3 = f.add_subplot(gs[3], sharey=ax2)
            if self.histogram_borders:
                grid(ax1.hist(x, bins=self.histogram_props[0], color=self.histogram_props[1], normed=True))
                grid(ax3.hist(y, bins=self.histogram_props[0], color=self.histogram_props[1], normed=True, orientation='horizontal'))
            elif self.boxplot_borders:
                grid(ax1.boxplot(x, vert=False, showmeans=True))
                grid(ax3.boxplot(y, vert=True, showmeans=True))
                if self.violin_plots:
                    ax1.violinplot(x, vert=False, showmedians=False, showmeans=False, showextrema=False)
                    ax3.violinplot(y, vert=True, showmedians=False, showmeans=False, showextrema=False)
            setp([ax1.get_xticklabels(), ax1.get_yticklabels(), ax3.get_xticklabels(), ax3.get_yticklabels()], visible=False)
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

    #TODO: Make sure the grid on the histogram borders is working properly

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
