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
from matplotlib.pyplot import show, subplot, yticks, xlabel, ylabel, figure, setp, savefig
from matplotlib.gridspec import GridSpec

# Numpy imports
from numpy import polyfit, polyval, sort, arange, array, linspace, mgrid, vstack, reshape

# Scipy imports
from scipy.stats import probplot, gaussian_kde

# local imports
from data.data import assign
from operations.data_operations import is_dict
# TODO: Add preferences back in a future version
# from ..preferences.preferences import GraphPreferences
# from six.moves import range


class MinimumSizeError(Exception):
    pass


class NoDataError(Exception):
    pass


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

        if 'intersect' in kwargs:
            x, y = assign(args[0], args[1])
            if x is None or y is None:
                raise NoDataError("Cannot graph because there is no data")
            try:
                x, y = x.data_prep(y)
            except TypeError:
                raise NoDataError("Cannot perform test because there is no data")
            if len(x) <= self._min_size or len(y) <= self._min_size:
                raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
            self._data = [x, y]
        else:
            data = list()
            for d in args:
                clean = assign(d).data_prep()
                if clean is None:
                    continue
                if len(clean) <= self._min_size:
                    raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
                data.append(clean)
            if len(data) < 1:
                raise NoDataError("Cannot perform test because there is no data")
            if len(data) == 1:
                data = data[0]
            self._data = data
        self.draw()

    @staticmethod
    def get_color(num):
        """Return a color based on the given num argument.

        :param num: An integer not equal to zero that returns a corresponding color
        :return: A color tuple calculated from the num argument
        """
        colors = [(0, 0.3, 0.7, 1),
                  (0, 0.5, 0, 1),
                  (1, 0, 0, 1),
                  (0, 1, 1, 1),
                  (1, 1, 0, 1),
                  (1, 0, 1, 1),
                  (1, 0.5, 0, 1),
                  (0.5, 0, 1, 1),
                  (0.5, 1, 0, 1),
                  (0, 0, 0, 1),
                  (0, 0, 1, 1)
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
        """Prepares and displays the graph based on the set class members."""
        pass


class GraphHisto(Graph):
    """Draws a histogram.

    New class members are bins, color and box_plot. The bins member is the number
    of histogram bins to draw. The color member is the color of the histogram area.
    The box_plot member is a boolean flag for whether to draw the corresponding
    box plot.
    """

    # nrows = 2
    # ncols = 1
    _xsize = 7
    _ysize = 6

    def __init__(self, data, **kwargs):
        """GraphHisto constructor.

        :param data: The data to be graphed. This arg sets the vector member.
        :param bins: The number of histogram bins to draw. This arg sets the bins member.
        :param name: The optional x-axis label.
        :param distribution: The theoretical distribution to fit.
        :param color: The optional color of the histogram as a formmated string.
        :param boxplot: Toggle the display of the optional boxplot.
        :param cdf: Toggle the display of the optional cumulative density function plot.
        :param violin_plot: Add a distribution density overlay to the boxplot.
        :param fit: Toggle the display of the best fit line for the specified distribution.
        :param save_to: Save the graph to the specified path
        :return: pass
        """
        self._bins = kwargs['bins'] if 'bins' in kwargs else 20
        self._distribution = kwargs['distribution'] if 'distribution' in kwargs else 'norm'
        self._color = kwargs['color'] if 'color' in kwargs else 'green'
        self._box_plot = kwargs['boxplot'] if 'boxplot' in kwargs else True
        self._cdf = kwargs['cdf'] if 'cdf' in kwargs else False
        self._violin_plot = kwargs['violin_plot'] if 'violin_plot' in kwargs else False
        self._histogram = kwargs['histogram'] if 'histogram' in kwargs else True
        self._fit = kwargs['fit'] if 'fit' in kwargs else False
        self._mean = kwargs['mean'] if 'mean' in kwargs else None
        self._std = kwargs['std_dev'] if 'std_dev' in kwargs else None
        self._sample = kwargs['sample'] if 'sample' in kwargs else True
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        yname = "Probability"
        name = 'Data'
        if 'name' in kwargs:
            name = kwargs['name']
        elif 'xname' in kwargs:
            name = kwargs['xname']

        super(GraphHisto, self).__init__(data, xname=name, yname=yname)
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
        distro_cdf = distro_class.cdf(distro, *parms)
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
        else:
            distro, distro_pdf, distro_cdf = None, None, None

        # Draw the cdf
        if self._cdf:
            x_cdf, y_cdf = self.calc_cdf()
            # ax_cdf = f.add_subplot(gs[0])
            ax_cdf = subplot(gs[0])
            ax_cdf.plot(x_cdf, y_cdf, 'k-')
            ax_cdf.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax_cdf.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            p.append(ax_cdf.get_xticklabels())
            if self._fit:
                ax_cdf.plot(distro, distro_cdf, 'r--', linewidth=2)
            yticks(arange(11) * 0.1)
            ylabel("Cumulative Probability")
        else:
            ax_cdf = None

        # Draw the box plot
        if self._box_plot:
            if self._cdf:
                # ax_box = f.add_subplot(gs[len(h_ratios)-2], sharex=ax_cdf)
                ax_box = subplot(gs[len(h_ratios) - 2], sharex=ax_cdf)
            else:
                # ax_box = f.add_subplot(gs[len(h_ratios)-2])
                ax_box = subplot(gs[len(h_ratios) - 2])
            # if GraphPreferences.distribution['violin']:
            bp = ax_box.boxplot(self._data, vert=False, showmeans=True)
            setp(bp['boxes'], color='k')
            setp(bp['whiskers'], color='k')
            vp = ax_box.violinplot(self._data, vert=False, showextrema=False, showmedians=False, showmeans=False)
            setp(vp['bodies'], facecolors=self.get_color(0))
            ax_box.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            # if GraphPreferences.distribution['boxplot']:
            yticks([])
            p.append(ax_box.get_xticklabels())
            # ax_hist = f.add_subplot(gs[len(h_ratios)-1], sharex=ax_box)
            ax_hist = subplot(gs[len(h_ratios) - 1], sharex=ax_box)
        else:
            # ax_hist = f.add_subplot(gs[len(h_ratios)-1])
            ax_hist = subplot(gs[len(h_ratios) - 1])

        # Draw the histogram
        ax_hist.hist(self._data, self._bins, normed=True, color=self.get_color(0))
        ax_hist.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        ax_hist.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        if self._fit:
            ax_hist.plot(distro, distro_pdf, 'r--', linewidth=2)
        if len(p) > 0:
            setp(p, visible=False)

        # set the labels and display the figure
        ylabel(self._yname)
        xlabel(self._xname)
        show()
        if self._save_to:
            savefig(self._save_to)
        pass


class GraphScatter(Graph):
    """Draws an x-by-y scatter plot.

    Unique class members are fit and style. The fit member is a boolean flag for
    whether to draw the linear best fit line. The style member is a tuple of
    formatted strings that set the matplotlib point style and line style. It is
    also worth noting that the vector member for the GraphScatter class is a
    tuple of xdata and ydata.
    """

    _nrows = 1
    _ncols = 1
    _xsize = 8
    _ysize = 7

    def __init__(self, xdata, ydata, **kwargs):
        """GraphScatter constructor.

        :param xdata: The x-axis data.
        :param ydata: The y-axis data.
        :param xname: The optional x-axis label.
        :param yname: The optional y-axis label.
        :param fit: Display the optional line fit.
        :return: pass
        """
        self._fit = kwargs['fit'] if 'fit' in kwargs else True
        self._points = kwargs['points'] if 'points' in kwargs else True
        self._contours = kwargs['contours'] if 'contours' in kwargs else False
        self._contour_props = (31, 1.1)
        # self.contour_props = tuple({'num_of_contours': 31, 'contour_width': 1.1}.values())
        self._histogram_props = (kwargs['bins'] if 'bins' in kwargs else 20, self.get_color(0))
        # self._histogram_borders = kwargs['histogram_borders'] if 'histogram_borders' in kwargs else False
        self._boxplot_borders = kwargs['boxplot_borders'] if 'boxplot_borders' in kwargs else False
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        yname = "y Data"
        xname = 'x Data'
        if 'yname' in kwargs:
            yname = kwargs['yname']
        elif 'xname' in kwargs:
            xname = kwargs['xname']
        super(GraphScatter, self).__init__(xdata, ydata, xname=xname, yname=yname, intersect=True)

    def calc_contours(self):
        xmin = self._data[0].min()
        xmax = self._data[0].max()
        ymin = self._data[1].min()
        ymax = self._data[1].max()

        values = vstack([self._data[0], self._data[1]])
        kernel = gaussian_kde(values)
        _x, _y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = vstack([_x.ravel(), _y.ravel()])
        _z = reshape(kernel(positions).T, _x.shape)
        return _x, _y, _z, arange(_z.min(), _z.max(), (_z.max() - _z.min()) / self._contour_props[0])

    def calc_fit(self):
        x = self._data[0]
        y = self._data[1]
        p = polyfit(x, y, 1, full=True)
        return polyval(p[0], x)

    def draw(self):
        x = self._data[0]
        y = self._data[1]
        h_ratio = [1, 1]
        w_ratio = [1, 1]
        # borders = self._histogram_borders or self._boxplot_borders

        # if borders:
        if self._boxplot_borders:
            self._nrows, self._ncols = 2, 2
            self._xsize, self._ysize = 7, 6
            h_ratio, w_ratio = [2, 5], [5, 2]
            main_plot = 2
        else:
            main_plot = 0
        f = figure(figsize=(self._xsize, self._ysize))
        f.suptitle('Bivariate', fontsize=14)
        # if borders:
        if self._boxplot_borders:
            gs = GridSpec(self._nrows, self._ncols, height_ratios=h_ratio, width_ratios=w_ratio, hspace=0, wspace=0)
        else:
            gs = GridSpec(self._nrows, self._ncols)
        # ax2 = f.add_subplot(gs[main_plot])
        ax2 = subplot(gs[main_plot])
        if self._points:
            # grid(ax2.plot(x, y, pointstyle, zorder=1))
            ax2.scatter(x, y, c=self.get_color(0), marker='o', linewidths=0, alpha=0.5, zorder=1)
            ax2.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax2.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        xlabel(self._xname)
        ylabel(self._yname)
        if self._contours:
            x_prime, y_prime, z, levels = self.calc_contours()
            ax2.contour(x_prime, y_prime, z, levels, linewidths=self._contour_props[1], nchunk=16,
                        extend='both', zorder=2)
        if self._fit:
            fit_line = self.calc_fit()
            ax2.plot([min(x), max(x)], [min(fit_line), max(fit_line)], 'r--', linewidth=2, zorder=3)
        # if borders:
        if self._boxplot_borders:
            # ax1 = f.add_subplot(gs[0], sharex=ax2)
            # ax3 = f.add_subplot(gs[3], sharey=ax2)
            ax1 = subplot(gs[0], sharex=ax2)
            ax3 = subplot(gs[3], sharey=ax2)
            # if self._histogram_borders:
            #     ax1.hist(x, bins=self._histogram_props[0], color=self._histogram_props[1], normed=True)
            #     ax3.hist(y, bins=self._histogram_props[0], color=self._histogram_props[1],
            #              normed=True, orientation='horizontal')
            # elif self._boxplot_borders:
            bpx = ax1.boxplot(x, vert=False, showmeans=True)
            bpy = ax3.boxplot(y, vert=True, showmeans=True)
            setp(bpx['boxes'], color='k')
            setp(bpx['whiskers'], color='k')
            setp(bpy['boxes'], color='k')
            setp(bpy['whiskers'], color='k')
            vpx = ax1.violinplot(x, vert=False, showmedians=False, showmeans=False, showextrema=False)
            vpy = ax3.violinplot(y, vert=True, showmedians=False, showmeans=False, showextrema=False)
            setp(vpx['bodies'], facecolors=self.get_color(0))
            setp(vpy['bodies'], facecolors=self.get_color(0))
            ax1.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax3.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            setp([ax1.get_xticklabels(), ax1.get_yticklabels(), ax3.get_xticklabels(), ax3.get_yticklabels()],
                 visible=False)
        show()
        if self._save_to:
            savefig(self._save_to)
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

    _nrows = 1
    _ncols = 2
    _xsize = 7.5
    _ysize = 5

    def __init__(self, *args, **kwargs):
        """GraphBoxplot constructor. NOTE: If vectors is a dict, the boxplots are
        graphed in random order instead of the provided order.

        :param vectors: A list of lists or dict of lists of the data to graph.
        :param groups: An optional list of boxplot labels. The order should match the order in vectors.
        :param xname: The optional x-axis label for the boxplots.
        :param yname: The optional y-axis label.
        :param nqp: Display the optional probability plot.
        :return: pass
        """
        xname = kwargs['xname'] if 'xname' in kwargs else 'Categories'
        yname = kwargs['yname'] if 'yname' in kwargs else 'Values'
        save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        self._nqp = kwargs['nqp'] if 'nqp' in kwargs else True
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        data = list()
        groups = list()
        if is_dict(args[0]):
            for g, d in args[0].items():
                data.append(d)
                groups.append(g)
            self._groups = groups
        else:
            self._groups = kwargs['groups'] if 'groups' in kwargs else list(range(1, len(args) + 1))
            if 'groups' in kwargs:
                if kwargs['groups']:
                    self._groups = kwargs['groups']
                else:
                    self._groups = list(range(1, len(args) + 1))
            else:
                self._groups = list(range(1, len(args) + 1))
            data = args
        self._prob = [probplot(d) for d in data] if self._nqp else list()
        super(GraphBoxplot, self).__init__(*data, xname=xname, yname=yname, save_to=save_to)

    def draw(self):
        if len(self._prob) > 0:
            self._xsize *= 2
            title = 'Oneway and Normal Quantile Plot'
        else:
            self._ncols = 1
            title = 'Oneway'
        f = figure(figsize=(self._xsize, self._ysize))
        f.suptitle(title, fontsize=14)
        gs = GridSpec(self._nrows, self._ncols, wspace=0)
        ax1 = subplot(gs[0])
        bp = ax1.boxplot(self._data, showmeans=True, labels=self._groups)
        setp(bp['boxes'], color='k')
        setp(bp['whiskers'], color='k')
        vp = ax1.violinplot(self._data, showextrema=False, showmedians=False, showmeans=False)
        for i in range(len(self._data)):
            setp(vp['bodies'][i], facecolors=self.get_color(i))
        ax1.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        ylabel(self._yname)
        xlabel(self._xname)
        if len(self._prob) > 0:
            ax2 = subplot(gs[1], sharey=ax1)
            # grid(which='major')
            for i, g in enumerate(self._prob):
                ax2.plot(g[0][0], g[0][1], marker='^', color=self.get_color(i), label=self._groups[i])
                ax2.plot(g[0][0], g[1][0] * g[0][0] + g[1][1], linestyle='--', linewidth=2, color=self.get_color(i))
            ax2.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax2.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax2.legend(loc='best')
            xlabel("Quantiles")
            setp(ax2.get_yticklabels(), visible=False)
        show()
        if self._save_to:
            savefig(self._save_to)
        pass
