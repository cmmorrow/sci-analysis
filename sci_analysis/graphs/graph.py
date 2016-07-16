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
from data.vector import Vector
from operations.data_operations import is_iterable, is_dict
# TODO: Add preferences back in a future version
# from ..preferences.preferences import GraphPreferences
# from six.moves import range


class Grid(object):

    def __init__(self, *args, **kwargs):
        self._nrows = int(kwargs['nrows']) if 'nrows' in kwargs else 1
        self._ncols = int(kwargs['ncols']) if 'ncols' in kwargs else 1
        self._xsize = int(kwargs['xsize']) if 'xsize' in kwargs else 7
        self._ysize = int(kwargs['ysize']) if 'ysize' in kwargs else 7
        self._title = kwargs['title'] if 'title' in kwargs else None
        self._yspace = kwargs['yspace'] if 'yspace' in kwargs else 0
        self._xspace = kwargs['xspace'] if 'xspace' in kwargs else 0
        self._graphs = list()
        self._axes = list()

        self._sub_x_size = [1]
        self._sub_y_size = [1]
        for obj in args:
            if isinstance(obj, Graph):
                self._graphs.append(obj)
                self._sub_x_size.insert(0, obj.xsub)
                self._sub_y_size.insert(0, obj.ysub)
            else:
                self._graphs.append(None)

        f = figure(figsize=(self._xsize, self._ysize))
        gs = GridSpec(self._nrows,
                      self._ncols,
                      height_ratios=self._sub_y_size,
                      width_ratios=self._sub_x_size,
                      hspace=self._yspace,
                      wspace=self._xspace)
        f.suptitle(self._title, fontsize=14)
        for i, graph in enumerate(self._graphs):
            if self._graphs[i]:
                self._axes.append(f.add_subplot(gs[i]))
            else:
                continue


class Graph(object):

    _xsub = 1
    _ysub = 1
    _min_size = 1

    def __init__(self, *args, **kwargs):
        self._xname = kwargs['xname'] if 'xname' in kwargs else 'x'
        self._yname = kwargs['yname'] if 'yname' in kwargs else 'y'
        self._display = kwargs['display'] if 'display' in kwargs else True
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None

        if self._display:
            self.draw()
            if self._save_to:
                savefig(self._save_to)

    def draw(self):
        """Prepares and displays the graph based on the set class members."""
        pass


class GraphCdf(Graph):

    _xsub = 1
    _ysub = 3

    def __init__(self, data, fit=False, distribution='norm', xname='data', yname='Probability', **kwargs):
        self._fit = fit
        self._distribution = distribution
        self._output = list()

        self._data = self.data_prep(data)
        x_cdf, y_cdf = self.calc_cdf()
        self._output.append((x_cdf, y_cdf, 'k-'))
        if self._fit:
            distro, distro_cdf = self.fit_distro(self._distribution)
            self._output.append((distro, distro_cdf, 'r--'))
        super(GraphCdf, self).__init__(data, xname, yname, **kwargs)

    def calc_cdf(self):
        x_sorted_vector = sort(self._data)
        if len(x_sorted_vector) == 0:
            return 0, 0
        y_sorted_vector = arange(len(x_sorted_vector) + 1) / float(len(x_sorted_vector))
        x_cdf = array([x_sorted_vector, x_sorted_vector]).T.flatten()
        y_cdf = array([y_sorted_vector[:(len(y_sorted_vector) - 1)], y_sorted_vector[1:]]).T.flatten()
        return x_cdf, y_cdf

    def fit_distro(self, distribution):
        distro_class = getattr(__import__('scipy.stats',
                                          globals(),
                                          locals(),
                                          [distribution], -1), distribution)
        parms = distro_class.fit(self._data)
        distro = linspace(distro_class.ppf(0.001, *parms), distro_class.ppf(0.999, *parms), 100)
        distro_cdf = distro_class.cdf(distro, *parms)
        return distro, distro_cdf

    def draw(self):
        for p in self._output:
            grid(plot(*p))
        xlabel(self._xname)
        ylabel(self._yname)


class GraphHistogram(Graph):

    _xsub = 1
    _ysub = 3

    def __init__(self, data, xname='Probability', yname='Data', distribution='norm',
                 fit=False, bins=20, color='green', **kwargs):
        self._fit = fit
        self._bins = bins
        self._color = color
        self._distribution = distribution
        self._output = list()

        self._data = self.data_prep(data)
        self._output.append((self._data, bins))
        if self._fit:
            distro, distro_pdf = self.fit_distro(self._distribution)
            self._output.append((distro, distro_pdf, 'r--'))
        super(GraphHistogram, self).__init__(data, xname, yname, **kwargs)

    def fit_distro(self, distribution):
        distro_class = getattr(__import__('scipy.stats',
                                          globals(),
                                          locals(),
                                          [distribution], -1), distribution)
        parms = distro_class.fit(self._data)
        distro = linspace(distro_class.ppf(0.001, *parms), distro_class.ppf(0.999, *parms), 100)
        distro_pdf = distro_class.pdf(distro, *parms)
        return distro, distro_pdf

    def draw(self):
        for p in self._output:
            grid(plot(*p))
        xlabel(self._xname)
        ylabel(self._yname)


class OldGraph(object):
    """The super class all other sci_analysis graphing classes descend from.
    Classes that descend from Graph should implement the draw method at bare minimum.

    Graph members are nrows, ncols, xsize, ysize, vector, xname and yname. The nrows
    member is the number of graphs that will span vertically. The ncols member is
    the number of graphs that will span horizontally. The xsize member is the horizontal
    size of the graph area. The ysize member is the vertical size of the graph area.
    The vector member the data to be plotted. The xname member is the x-axis label.
    The yname member is the y-axis label.

    """

    _nrows = 1
    _ncols = 1
    _xsize = 5
    _ysize = 5
    _min_size = 1

    def __init__(self, *args, **kwargs):
        """Converts the data argument to a Vector object and sets it to the Graph
        object's vector member. Sets the xname and yname arguments as the axis
        labels. The default values are "x" and "y".

        :param data: The data to plot
        :param xname: The x-axis label
        :param yname: The y-axis label
        :param save_to: Save the graph to the specified path
        :return: pass
        """

        self._xname = kwargs['xname'] if 'xname' in kwargs else 'x'
        self._yname = kwargs['yname'] if 'yname' in kwargs else 'y'
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else 'save_to'

        data = self.data_prep(args)
        if len(data) <= 1:
            try:
                data = data[0]
            except IndexError:
                raise EmptyVectorError("Passed data is empty")
        self._data = data
        if self._save_to:
            savefig(self._save_to)
        self.draw()

    def data_prep(self, data):
        clean_list = list()
        for d in data:
            if not is_iterable(d):
                try:
                    clean_list.append(float(d))
                except (ValueError, TypeError):
                    continue
            else:
                v = drop_nan(d) if is_vector(d) else drop_nan(Vector(d))
                if not v:
                    continue
                if len(v) <= self._min_size:
                    raise MinimumSizeError("length of data is less than the minimum size {}"
                                           .format(self._min_size))
                clean_list.append(v)
        return clean_list
    
    def draw(self):
        """Prepares and displays the graph based on the set class members."""
        pass


class GraphHisto(OldGraph):
    """Draws a histogram.

    New class members are bins, color and box_plot. The bins member is the number
    of histogram bins to draw. The color member is the color of the histogram area.
    The box_plot member is a boolean flag for whether to draw the corresponding
    box plot.
    """

    # nrows = 2
    # ncols = 1
    _ysize = 4

    def __init__(self, data, **kwargs):
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
        self._bins = kwargs['bins'] if 'bins' in kwargs else 20
        self._distribution = kwargs['distribution'] if 'distribution' in kwargs else 'norm'
        self._color = kwargs['color'] if 'color' in kwargs else 'green'
        self._box_plot = kwargs['box_plot'] if 'box_plot' in kwargs else True
        self._cdf = kwargs['cdf'] if 'cdf' in kwargs else False
        self._violin_plot = kwargs['violin_plot'] if 'violin_plot' in kwargs else False
        self._histogram = kwargs['histogram'] if 'histogram' in kwargs else True
        self._fit = kwargs['fit'] if 'fit' in kwargs else False
        self._mean = kwargs['mean'] if 'mean' in kwargs else None
        self._std = kwargs['std_dev'] if 'std_dev' in kwargs else None
        self._sample = kwargs['sample'] if 'sample' in kwargs else True
        self._yname = "Probability"
        self._name = 'Data'
        if 'name' in kwargs:
            self._name = kwargs['name']
        elif 'xname' in kwargs:
            self._name = kwargs['xname']

        super(GraphHisto, self).__init__(data)
        # if GraphPreferences.Plot.boxplot != GraphPreferences.Plot.defaults[0]:
        #    _box_plot = GraphPreferences.Plot.boxplot
        # else:
        #    _box_plot = box_plot
        # if GraphPreferences.Plot.cdf != GraphPreferences.Plot.defaults[2]:
        #    _cdf = GraphPreferences.Plot.cdf
        # else:
        #    _cdf = cdf

    def fit_distro(self):
        distro_class = getattr(__import__('scipy.stats',
                                          globals(),
                                          locals(),
                                          [self._distribution], -1), self._distribution)
        parms = distro_class.fit(self._data)
        distro = linspace(distro_class.ppf(0.001, *parms), distro_class.ppf(0.999, *parms), 100)
        distro_pdf = distro_class.pdf(distro, *parms)
        if self._fit:
            distro_cdf = distro_class.cdf(distro, *parms)
        else:
            distro_cdf = None
        return distro, distro_pdf, distro_cdf

    def calc_cdf(self):
        x_sorted_vector = sort(self._data)
        if len(x_sorted_vector) == 0:
            return 0, 0
        y_sorted_vector = arange(len(x_sorted_vector) + 1) / float(len(x_sorted_vector))
        x_cdf = array([x_sorted_vector, x_sorted_vector]).T.flatten()
        y_cdf = array([y_sorted_vector[:(len(y_sorted_vector)-1)], y_sorted_vector[1:]]).T.flatten()
        return x_cdf, y_cdf

    def draw(self):
        # Setup the grid variables
        histo_span = 3
        box_plot_span = 1
        cdf_span = 3
        h_ratios = [histo_span]
        p = []
        if self._box_plot:
            self._ysize += 1
            self._nrows += 1
            h_ratios.insert(0, box_plot_span)
        if self._cdf:
            self._ysize += 4
            self._nrows += 1
            h_ratios.insert(0, cdf_span)

        # Create the figure and grid spec
        f = figure(figsize=(self._xsize, self._ysize))
        gs = GridSpec(self._nrows, self._ncols, height_ratios=h_ratios, hspace=0)

        # Set the title
        title = "Distribution"
        if self._mean is not None and self._std is not None:
            if self._sample:
                title = r"{}{}$\bar x = {},  s = {}$".format(title, "\n", self._mean, self._std)
            else:
                title = r"{}{}$\mu = {}$,  $\sigma = {}$".format(title, "\n", self._mean, self._std)
        f.suptitle(title, fontsize=14)

        # Adjust the bin size if it's greater than the vector size
        if len(self._data) < self._bins:
            self._bins = len(self._data)

        # Fit the distribution
        if self._fit:
            distro, distro_pdf, distro_cdf = self.fit_distro()

        # Draw the cdf
        if self._cdf:
            x_cdf, y_cdf = self.calc_cdf()
            ax_cdf = f.add_subplot(gs[0])
            grid(ax_cdf.plot(x_cdf, y_cdf, 'k-'))
            p.append(ax_cdf.get_xticklabels())
            if self._fit:
                ax_cdf.plot(distro, distro_cdf, 'r--', linewidth=2)
            yticks(arange(11) * 0.1)
            ylabel("Cumulative Probability")

        # Draw the box plot
        if self._box_plot or self._violin_plot:
            if self._cdf:
                ax_box = f.add_subplot(gs[len(h_ratios)-2], sharex=ax_cdf)
            else:
                ax_box = f.add_subplot(gs[len(h_ratios)-2])
            # if GraphPreferences.distribution['violin']:
            if self._violin_plot:
                grid(ax_box.violinplot(self._data.data, vert=False, showextrema=False, showmedians=False, showmeans=False))
            # if GraphPreferences.distribution['boxplot']:
            grid(ax_box.boxplot(self._data.data, vert=False, showmeans=True))
            yticks([])
            p.append(ax_box.get_xticklabels())
            ax_hist = f.add_subplot(gs[len(h_ratios)-1], sharex=ax_box)
        else:
            ax_hist = f.add_subplot(gs[len(h_ratios)-1])

        # Draw the histogram
        grid(ax_hist.hist(self._data.data, self._bins, normed=True, color=self._color))
        if self._fit:
            ax_hist.plot(distro, distro_pdf, 'r--', linewidth=2)
        if len(p) > 0:
            setp(p, visible=False)

        # set the labels and display the figure
        ylabel("".format(self._yname))
        xlabel(self._xname)
        show()
        if self._save_to:
            savefig(self._save_to)
        pass


class GraphScatter(OldGraph):
    """Draws an x-by-y scatter plot.

    Unique class members are fit and style. The fit member is a boolean flag for
    whether to draw the linear best fit line. The style member is a tuple of
    formatted strings that set the matplotlib point style and line style. It is
    also worth noting that the vector member for the GraphScatter class is a
    tuple of xdata and ydata.
    """

    _nrows = 1
    _ncols = 1
    _xsize = 5
    _ysize = 5

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
        xmin = self._data[0].data.min()
        xmax = self._data[0].data.max()
        ymin = self._data[1].data.min()
        ymax = self._data[1].data.max()

        values = vstack([self._data[0].data, self._data[1].data])
        kernel = gaussian_kde(values)
        _x, _y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = vstack([_x.ravel(), _y.ravel()])
        _z = reshape(kernel(positions).T, _x.shape)
        return _x, _y, _z, arange(_z.min(), _z.max(), (_z.max() - _z.min()) / self.contour_props[0])

    def calc_fit(self):
        x = self._data[0].data
        y = self._data[1].data
        p = polyfit(x, y, 1, full=True)
        return polyval(p[0], x)

    def draw(self):
        x = self._data[0].data
        y = self._data[1].data
        pointstyle = self.style[0]
        linestyle = self.style[1]
        h_ratio = [1, 1]
        w_ratio = [1, 1]
        borders = self.histogram_borders or self.boxplot_borders

        if borders:
            self._nrows, self._ncols = 2, 2
            self._xsize, self._ysize = 7, 6
            h_ratio, w_ratio = [2, 5], [5, 2]
            main_plot = 2
        else:
            main_plot = 0
        f = figure(figsize=(self._xsize, self._ysize))
        if borders:
            gs = GridSpec(self._nrows, self._ncols, height_ratios=h_ratio, width_ratios=w_ratio, hspace=0.1, wspace=0.1)
        else:
            gs = GridSpec(self._nrows, self._ncols)
        ax2 = f.add_subplot(gs[main_plot])
        if self.points:
            grid(ax2.plot(x, y, pointstyle, zorder=1))
        xlabel(self._xname)
        ylabel(self._yname)
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


class GraphBoxplot(OldGraph):
    """Draws box plots of the provided data as well as an optional probability plot.

    Unique class members are groups, nqp and prob. The groups member is a list of
    labels for each boxplot. If groups is an empty list, sequentially ascending
    numbers are used for each boxplot. The nqp member is a flag that turns the
    probability plot on or off. The prob member is a list of tuples that contains
    the data used to graph the probability plot. It is also worth noting that the
    vector member for the GraphBoxplot is a list of lists that contain the data
    for each boxplot.
    """

    _nrows = 1
    _ncols = 2
    _xsize = 7.5
    _ysize = 5

    # TODO: Make sure the grid on the histogram borders is working properly

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
            groups = list(range(1, len(self._data) + 1))
        for i, v in enumerate(self._data):
            self._data[i] = v = drop_nan(v)
            if len(v) == 0:
                groups = groups[:i] + groups[i + 1:]
                continue
            if nqp:
                q, fit = probplot(v)
                self.prob.append((q, fit))
        self.groups = groups
        self.draw()

    @staticmethod
    def get_color(num):
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
            figure(figsize=(self._xsize * 2, self._ysize))
            ax2 = subplot(self._nrows, self._ncols, 2)
            grid(which='major')
            for i, g in enumerate(self.prob):
                plot(g[0][0], g[0][1], marker='^', color=self.get_color(i), label=self.groups[i])
                plot(g[0][0], g[1][0] * g[0][0] + g[1][1], linestyle='-', color=self.get_color(i))
            legend(loc='best')
            xlabel("Quantiles")
            subplot(self._nrows, self._ncols, 1, sharey=ax2)
        else:
            figure(figsize=(self._xsize, self._ysize))
        grid(boxplot(self._data, showmeans=True, labels=self.groups))
        ylabel(self._yname)
        xlabel(self._xname)
        show()
        pass
