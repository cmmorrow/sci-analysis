import warnings
import six
from math import sqrt, fabs

# matplotlib imports
from matplotlib.pyplot import (
    show, subplot, yticks, xlabel, ylabel, figure, setp, savefig, close, xticks, subplots_adjust
)
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# Numpy imports
from numpy import (
    polyfit, polyval, sort, arange, array, linspace, mgrid, vstack, std, sum, mean, median
)

# Scipy imports
from scipy.stats import probplot, gaussian_kde, t

# local imports
from .base import Graph
from ..data import Vector, is_dict, is_group, is_vector
from ..analysis.exc import NoDataError


def future(message):
    warnings.warn(message, FutureWarning, stacklevel=2)


class VectorGraph(Graph):

    def __init__(self, sequence, **kwargs):
        """Converts the data argument to a Vector object and sets it to the Graph
        object's vector member. Sets the xname and yname arguments as the axis
        labels. The default values are "x" and "y".
        """

        if is_vector(sequence):
            super(VectorGraph, self).__init__(sequence, **kwargs)
        else:
            super(VectorGraph, self).__init__(Vector(sequence), **kwargs)
        if len(self._data.groups.keys()) == 0:
            raise NoDataError("Cannot draw graph because there is no data.")
        self.draw()

    def draw(self):
        """
        Prepares and displays the graph based on the set class members.
        """
        raise NotImplementedError


class GraphHisto(VectorGraph):
    """Draws a histogram.

    New class members are bins, color and box_plot. The bins member is the number
    of histogram bins to draw. The color member is the color of the histogram area.
    The box_plot member is a boolean flag for whether to draw the corresponding
    box plot.
    """

    _xsize = 5
    _ysize = 4

    def __init__(self, data, **kwargs):
        """GraphHisto constructor.

        :param data: The data to be graphed.
        :param _bins: The number of histogram bins to draw. This arg sets the bins member.
        :param _name: The optional x-axis label.
        :param _distribution: The theoretical distribution to fit.
        :param _box_plot: Toggle the display of the optional boxplot.
        :param _cdf: Toggle the display of the optional cumulative density function plot.
        :param _fit: Toggle the display of the best fit line for the specified distribution.
        :param _mean: The mean to be displayed on the graph title.
        :param _std: The standard deviation to be displayed on the graph title.
        :param _sample: Sets x-bar and s if true, else mu and sigma for displaying on the graph title.
        :param _title: The title of the graph.
        :param _save_to: Save the graph to the specified path.
        :return: pass
        """
        self._bins = kwargs.get('bins', 20)
        self._distribution = kwargs.get('distribution', 'norm')
        self._box_plot = kwargs.get('boxplot', True)
        self._cdf = kwargs.get('cdf', False)
        self._fit = kwargs.get('fit', False)
        self._mean = kwargs.get('mean')
        self._std = kwargs.get('std_dev')
        self._sample = kwargs.get('sample', False)
        self._title = kwargs.get('title', 'Distribution')
        self._save_to = kwargs.get('save_to')
        yname = kwargs.get('yname', 'Probability')
        name = kwargs.get('name') or kwargs.get('xname') or 'Data'

        super(GraphHisto, self).__init__(data, xname=name, yname=yname)

    def fit_distro(self):
        """
        Calculate the fit points for a specified distribution.

        Returns
        -------
        fit_parms : tuple
            First value - The x-axis points
            Second value - The pdf y-axis points
            Third value - The cdf y-axis points

        """
        distro_class = getattr(
            __import__(
                'scipy.stats',
                globals(),
                locals(),
                [self._distribution],
                0,
            ),
            self._distribution
        )
        parms = distro_class.fit(self._data.data)
        distro = linspace(distro_class.ppf(0.001, *parms), distro_class.ppf(0.999, *parms), 100)
        distro_pdf = distro_class.pdf(distro, *parms)
        distro_cdf = distro_class.cdf(distro, *parms)
        return distro, distro_pdf, distro_cdf

    def calc_cdf(self):
        """
        Calcuate the cdf points.

        Returns
        -------
        coordinates : tuple
            First value - The cdf x-axis points
            Second value - The cdf y-axis points
        """
        x_sorted_vector = sort(self._data.data)
        if len(x_sorted_vector) == 0:
            return 0, 0
        y_sorted_vector = arange(len(x_sorted_vector) + 1) / float(len(x_sorted_vector))
        x_cdf = array([x_sorted_vector, x_sorted_vector]).T.flatten()
        y_cdf = array([y_sorted_vector[:(len(y_sorted_vector)-1)], y_sorted_vector[1:]]).T.flatten()
        return x_cdf, y_cdf

    def draw(self):
        """
        Draws the histogram based on the set parameters.

        Returns
        -------
        pass
        """
        # Setup the grid variables
        histo_span = 3
        box_plot_span = 1
        cdf_span = 3
        h_ratios = [histo_span]
        p = []
        if self._box_plot:
            self._ysize += 0.5
            self._nrows += 1
            h_ratios.insert(0, box_plot_span)
        if self._cdf:
            self._ysize += 2
            self._nrows += 1
            h_ratios.insert(0, cdf_span)

        # Create the figure and grid spec
        f = figure(figsize=(self._xsize, self._ysize))
        gs = GridSpec(self._nrows, self._ncols, height_ratios=h_ratios, hspace=0)

        # Set the title
        title = self._title
        if self._mean and self._std:
            if self._sample:
                title = r"{}{}$\bar x = {:.4f},  s = {:.4f}$".format(title, "\n", self._mean, self._std)
            else:
                title = r"{}{}$\mu = {:.4f}$,  $\sigma = {:.4f}$".format(title, "\n", self._mean, self._std)
        f.suptitle(title, fontsize=14)

        # Adjust the bin size if it's greater than the vector size
        if len(self._data.data) < self._bins:
            self._bins = len(self._data.data)

        # Fit the distribution
        if self._fit:
            distro, distro_pdf, distro_cdf = self.fit_distro()
        else:
            distro, distro_pdf, distro_cdf = None, None, None

        # Draw the cdf
        if self._cdf:
            x_cdf, y_cdf = self.calc_cdf()
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
                ax_box = subplot(gs[len(h_ratios) - 2], sharex=ax_cdf)
            else:
                ax_box = subplot(gs[len(h_ratios) - 2])
            bp = ax_box.boxplot(self._data.data, vert=False, showmeans=True)
            setp(bp['boxes'], color='k')
            setp(bp['whiskers'], color='k')
            vp = ax_box.violinplot(self._data.data, vert=False, showextrema=False, showmedians=False, showmeans=False)
            setp(vp['bodies'], facecolors=self.get_color(0))
            ax_box.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            yticks([])
            p.append(ax_box.get_xticklabels())
            ax_hist = subplot(gs[len(h_ratios) - 1], sharex=ax_box)
        else:
            ax_hist = subplot(gs[len(h_ratios) - 1])

        # Draw the histogram
        # First try to use the density arg which replaced normed (which is now depricated) in matplotlib 2.2.2
        try:
            ax_hist.hist(self._data.data, self._bins, density=True, color=self.get_color(0), zorder=0)
        except TypeError:
            ax_hist.hist(self._data.data, self._bins, normed=True, color=self.get_color(0), zorder=0)
        ax_hist.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        ax_hist.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        if self._fit:
            ax_hist.plot(distro, distro_pdf, 'r--', linewidth=2)
        if len(p) > 0:
            setp(p, visible=False)

        # set the labels and display the figure
        ylabel(self._yname)
        xlabel(self._xname)
        if self._save_to:
            savefig(self._save_to)
            close(f)
        else:
            show()
        pass


class GraphScatter(VectorGraph):
    """Draws an x-by-y scatter plot.

    Unique class members are fit and style. The fit member is a boolean flag for
    whether to draw the linear best fit line. The style member is a tuple of
    formatted strings that set the matplotlib point style and line style. It is
    also worth noting that the vector member for the GraphScatter class is a
    tuple of xdata and ydata.
    """

    _nrows = 1
    _ncols = 1
    _xsize = 6
    _ysize = 5

    def __init__(self, xdata, ydata=None, **kwargs):
        """GraphScatter constructor.

        :param xdata: The x-axis data.
        :param ydata: The y-axis data.
        :param fit: Display the optional line fit.
        :param points: Display the scatter points.
        :param contours: Display the density contours
        :param boxplot_borders: Display the boxplot borders
        :param highlight: an array-like with points to highlight based on labels
        :param labels: a vector object with the graph labels
        :param title: The title of the graph.
        :param save_to: Save the graph to the specified path.
        :return: pass
        """
        self._fit = kwargs.get('fit', True)
        self._points = kwargs.get('points', True)
        self._labels = kwargs.get('labels', None)
        self._highlight = kwargs.get('highlight', None)
        self._contours = kwargs.get('contours', False)
        self._contour_props = (31, 1.1)
        self._boxplot_borders = kwargs.get('boxplot_borders', False)
        self._title = kwargs['title'] if 'title' in kwargs else 'Bivariate'
        self._save_to = kwargs.get('save_to', None)
        yname = kwargs.get('yname', 'y Data')
        xname = kwargs.get('xname', 'x Data')
        if ydata is None:
            if is_vector(xdata):
                super(GraphScatter, self).__init__(xdata, xname=xname, yname=yname)
            else:
                raise AttributeError('ydata argument cannot be None.')
        else:
            super(GraphScatter, self).__init__(
                Vector(xdata, other=ydata, labels=self._labels),
                xname=xname,
                yname=yname,
            )

    def calc_contours(self):
        """
        Calculates the density contours.

        Returns
        -------
        contour_parms : tuple
            First value - x-axis points
            Second value - y-axis points
            Third value - z-axis points
            Fourth value - The contour levels
        """
        xmin = self._data.data.min()
        xmax = self._data.data.max()
        ymin = self._data.other.min()
        ymax = self._data.other.max()

        values = vstack([self._data.data, self._data.other])
        kernel = gaussian_kde(values)
        _x, _y = mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = vstack([_x.ravel(), _y.ravel()])
        _z = kernel.evaluate(positions).T.reshape(_x.shape)
        return _x, _y, _z, arange(_z.min(), _z.max(), (_z.max() - _z.min()) / self._contour_props[0])

    def calc_fit(self):
        """
        Calculates the best fit line using sum of squares.

        Returns
        -------
        fit_coordinates : list
            A list of the min and max fit points.
        """
        x = self._data.data
        y = self._data.other
        p = polyfit(x, y, 1)
        fit = polyval(p, x)
        if p[0] > 0:
            return (x.min(), x.max()), (fit.min(), fit.max())
        else:
            return (x.min(), x.max()), (fit.max(), fit.min())

    def draw(self):
        """
        Draws the scatter plot based on the set parameters.

        Returns
        -------
        pass
        """
        
        # Setup the grid variables
        x = self._data.data
        y = self._data.other
        h_ratio = [1, 1]
        w_ratio = [1, 1]

        # Setup the figure and gridspec
        if self._boxplot_borders:
            self._nrows, self._ncols = 2, 2
            self._xsize = self._xsize + 0.5
            self._ysize = self._ysize + 0.5
            h_ratio, w_ratio = (1.5, 5.5), (5.5, 1.5)
            main_plot = 2
        else:
            main_plot = 0

        # Setup the figure
        f = figure(figsize=(self._xsize, self._ysize))
        f.suptitle(self._title, fontsize=14)
        if self._boxplot_borders:
            gs = GridSpec(self._nrows, self._ncols, height_ratios=h_ratio, width_ratios=w_ratio, hspace=0, wspace=0)
        else:
            gs = GridSpec(self._nrows, self._ncols)

        ax1 = None
        ax3 = None

        # Draw the boxplot borders
        if self._boxplot_borders:
            ax1 = subplot(gs[0])
            ax3 = subplot(gs[3])
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
            setp(
                [
                    ax1.get_xticklabels(), ax1.get_yticklabels(), ax3.get_xticklabels(), ax3.get_yticklabels()
                ], visible=False
            )

        # Draw the main graph
        ax2 = subplot(gs[main_plot], sharex=ax1, sharey=ax3)

        # Draw the points
        if self._points:
            # A 2-D array needs to be passed to prevent matplotlib from applying the default cmap if the size < 4.
            color = (self.get_color(0),)
            alpha_trans = 0.7
            if self._highlight is not None:
                # Find index of the labels which are in the highlight list
                labelmask = self._data.labels.isin(self._highlight)

                # Get x and y position of those labels
                x_labels = x.loc[labelmask]
                y_labels = y.loc[labelmask]
                x_nolabels = x.loc[~labelmask]
                y_nolabels = y.loc[~labelmask]
                ax2.scatter(x_labels, y_labels, c=color, marker='o', linewidths=0, alpha=alpha_trans, zorder=1)
                ax2.scatter(x_nolabels, y_nolabels, c=color, marker='o', linewidths=0, alpha=.2, zorder=1)
                for k in self._data.labels[labelmask].index:
                    ax2.annotate(self._data.labels[k], xy=(x[k], y[k]), alpha=1, color=color[0])
            else:
                ax2.scatter(x, y, c=color, marker='o', linewidths=0, alpha=alpha_trans, zorder=1)

        # Draw the contours
        if self._contours:
            x_prime, y_prime, z, levels = self.calc_contours()
            ax2.contour(x_prime, y_prime, z, levels, linewidths=self._contour_props[1], nchunk=16,
                        extend='both', zorder=2)

        # Draw the fit line
        if self._fit:
            fit_x, fit_y = self.calc_fit()
            ax2.plot(fit_x, fit_y, 'r--', linewidth=2, zorder=3)

        # Draw the grid lines and labels
        ax2.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        ax2.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        xlabel(self._xname)
        ylabel(self._yname)

        # Save the figure to disk or display
        if self._save_to:
            savefig(self._save_to)
            close(f)
        else:
            show()
        pass


class GraphGroupScatter(VectorGraph):
    """Draws an x-by-y scatter plot with more than a single group.

    Unique class members are fit and style. The fit member is a boolean flag for
    whether to draw the linear best fit line. The style member is a tuple of
    formatted strings that set the matplotlib point style and line style. It is
    also worth noting that the vector member for the GraphScatter class is a
    tuple of xdata and ydata.
    """

    _nrows = 1
    _ncols = 1
    _xsize = 6
    _ysize = 5

    def __init__(self, xdata, ydata=None, groups=None, **kwargs):
        """GraphScatter constructor.

        :param xdata: The x-axis data.
        :param ydata: The y-axis data.
        :param _fit: Display the optional line fit.
        :param _highlight: Give list of groups to highlight in scatter.
        :param _points: Display the scatter points.
        :param _contours: Display the density contours
        :param _boxplot_borders: Display the boxplot borders
        :param _labels: a vector object with the graph labels
        :param _title: The title of the graph.
        :param _save_to: Save the graph to the specified path.
        :return: pass
        """
        self._fit = kwargs['fit'] if 'fit' in kwargs else True
        self._points = kwargs['points'] if 'points' in kwargs else True
        self._labels = kwargs['labels'] if 'labels' in kwargs else None
        self._highlight = kwargs['highlight'] if 'highlight' in kwargs else None
        self._boxplot_borders = kwargs['boxplot_borders'] if 'boxplot_borders' in kwargs else True
        self._title = kwargs['title'] if 'title' in kwargs else 'Group Bivariate'
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        yname = kwargs['yname'] if 'yname' in kwargs else 'y Data'
        xname = kwargs['xname'] if 'xname' in kwargs else 'x Data'
        if ydata is None:
            if is_vector(xdata):
                super(GraphGroupScatter, self).__init__(xdata, xname=xname, yname=yname)
            else:
                raise AttributeError('ydata argument cannot be None.')
        else:
            super(GraphGroupScatter, self).__init__(Vector(
                xdata,
                other=ydata,
                groups=groups,
                labels=self._labels
            ), xname=xname, yname=yname)

    @staticmethod
    def calc_fit(x, y):
        """
        Calculates the best fit line using sum of squares.

        Returns
        -------
        fit_coordinates : list
            A list of the min and max fit points.
        """
        p = polyfit(x, y, 1)
        fit = polyval(p, x)
        if p[0] > 0:
            return (x.min(), x.max()), (fit.min(), fit.max())
        else:
            return (x.min(), x.max()), (fit.max(), fit.min())

    def draw(self):
        """
        Draws the scatter plot based on the set parameters.

        Returns
        -------
        pass
        """
        
        # Setup the grid variables
        x = self._data.data
        y = self._data.other
        groups = sorted(self._data.groups.keys())
        h_ratio = [1, 1]
        w_ratio = [1, 1]

        # Setup the figure and gridspec
        if self._boxplot_borders:
            self._nrows, self._ncols = 2, 2
            self._xsize = self._xsize + 0.5
            self._ysize = self._ysize + 0.5
            h_ratio, w_ratio = (1.5, 5.5), (5.5, 1.5)
            main_plot = 2
        else:
            main_plot = 0

        # Setup the figure
        f = figure(figsize=(self._xsize, self._ysize))
        f.suptitle(self._title, fontsize=14)
        if self._boxplot_borders:
            gs = GridSpec(self._nrows, self._ncols, height_ratios=h_ratio, width_ratios=w_ratio, hspace=0, wspace=0)
        else:
            gs = GridSpec(self._nrows, self._ncols)

        ax1 = None
        ax3 = None

        # Draw the boxplot borders
        if self._boxplot_borders:
            ax1 = subplot(gs[0])
            ax3 = subplot(gs[3])
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

        # Draw the main graph
        ax2 = subplot(gs[main_plot], sharex=ax1, sharey=ax3)

        for grp, (grp_x, grp_y) in self._data.paired_groups.items():
            i = groups.index(grp)
            alpha_trans = 0.65
            if self._highlight is not None:
                try:
                    if grp not in self._highlight:
                        alpha_trans = 0.2
                except TypeError:
                    pass
            if isinstance(grp, six.string_types) and len(grp) > 20:
                grp = grp[0:21] + '...'

            # Draw the points
            if self._points:
                # A 2-D array needs to be passed to prevent matplotlib from applying the default cmap if the size < 4.
                color = (self.get_color(i),)
                scatter_kwargs = dict(
                    c=color,
                    marker='o',
                    linewidths=0,
                    zorder=1,
                )

                # Draw the point labels
                if self._data.has_labels and self._highlight is not None:

                    # If a group is in highlights and labels are also given
                    if grp in self._highlight:
                        scatter_kwargs.update(
                            dict(
                                alpha=alpha_trans,
                                label=grp
                            )
                        )
                        ax2.scatter(grp_x, grp_y, **scatter_kwargs)
                    # Highlight the specified labels
                    else:
                        labelmask = self._data.group_labels[grp].isin(self._highlight)
                        # Get x and y position of those labels
                        x_labels = grp_x.loc[labelmask]
                        y_labels = grp_y.loc[labelmask]
                        x_nolabels = grp_x.loc[~labelmask]
                        y_nolabels = grp_y.loc[~labelmask]
                        scatter_kwargs.update(
                            dict(
                                alpha=0.65,
                                label=grp if any(labelmask) else None,
                            )
                        )
                        ax2.scatter(x_labels, y_labels, **scatter_kwargs)
                        scatter_kwargs.update(
                            dict(
                                alpha=0.2,
                                label=None if any(labelmask) else grp,
                            )
                        )
                        ax2.scatter(x_nolabels, y_nolabels, **scatter_kwargs)
                        # Add the annotations
                        for k in self._data.group_labels[grp][labelmask].index:
                            clr = color[0]
                            ax2.annotate(self._data.group_labels[grp][k], xy=(grp_x[k], grp_y[k]), alpha=1, color=clr)
                else:
                    scatter_kwargs.update(
                        dict(
                            alpha=alpha_trans,
                            label=grp,
                        )
                    )
                    ax2.scatter(grp_x, grp_y, **scatter_kwargs)

            # Draw the fit line
            if self._fit:
                fit_x, fit_y = self.calc_fit(grp_x, grp_y)
                if self._points:
                    ax2.plot(fit_x, fit_y, linestyle='--', color=self.get_color(i), linewidth=2, zorder=2)
                else:
                    ax2.plot(fit_x, fit_y, linestyle='--', color=self.get_color(i), linewidth=2, zorder=2, label=grp)

        # Draw the legend
        if (self._fit or self._points) and len(groups) > 1:
            ax2.legend(loc='best')

        # Draw the grid lines and labels
        ax2.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        ax2.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        xlabel(self._xname)
        ylabel(self._yname)

        # Save the figure to disk or display
        if self._save_to:
            savefig(self._save_to)
            close(f)
        else:
            show()
        pass


class GraphBoxplot(VectorGraph):
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
    _ncols = 1
    _xsize = 5.75
    _ysize = 5
    _default_alpha = 0.05

    def __init__(self, *args, **kwargs):
        """GraphBoxplot constructor. NOTE: If vectors is a dict, the boxplots are
        graphed in random order instead of the provided order.

        :param groups: An optional list of boxplot labels. The order should match the order in vectors.
        :param nqp: Display the optional probability plot.
        :param _title: The title of the graph.
        :param _save_to: Save the graph to the specified path.
        :return: pass
        """
        name = kwargs['name'] if 'name' in kwargs else 'Values'
        categories = kwargs['categories'] if 'categories' in kwargs else 'Categories'
        xname = kwargs['xname'] if 'xname' in kwargs else categories
        yname = kwargs['yname'] if 'yname' in kwargs else name
        self._title = kwargs['title'] if 'title' in kwargs else 'Oneway'
        self._nqp = kwargs['nqp'] if 'nqp' in kwargs else True
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        self._gmean = kwargs['gmean'] if 'gmean' in kwargs else True
        self._gmedian = kwargs['gmedian'] if 'gmedian' in kwargs else True
        self._circles = kwargs['circles'] if 'circles' in kwargs else True
        self._alpha = kwargs['alpha'] if 'alpha' in kwargs else self._default_alpha
        if 'title' in kwargs:
            self._title = kwargs['title']
        elif self._nqp:
            self._title = 'Oneway and Normal Quantile Plot'
        else:
            self._title = 'Oneway'

        if is_vector(args[0]):
            data = args[0]
        elif is_dict(args[0]):
            data = Vector()
            for g, d in args[0].items():
                data.append(Vector(d, groups=[g] * len(d)))
        else:
            if is_group(args) and len(args) > 1:
                future('Graphing boxplots by passing multiple arguments will be removed in a future version. '
                       'Instead, pass unstacked arguments as a dictionary.')
                data = Vector()
                if 'groups' in kwargs:
                    if len(kwargs['groups']) != len(args):
                        raise AttributeError('The length of passed groups does not match the number passed data.')
                    for g, d in zip(kwargs['groups'], args):
                        data.append(Vector(d, groups=[g] * len(d)))
                else:
                    for d in args:
                        data.append(Vector(d))
            else:
                if 'groups' in kwargs:
                    if len(kwargs['groups']) != len(args[0]):
                        raise AttributeError('The length of passed groups does not match the number passed data.')
                    data = Vector(args[0], groups=kwargs['groups'])
                else:
                    data = Vector(args[0])
        super(GraphBoxplot, self).__init__(data, xname=xname, yname=yname, save_to=self._save_to)

    @staticmethod
    def grand_mean(data):
        return mean([mean(sample) for sample in data])

    @staticmethod
    def grand_median(data):
        return median([median(sample) for sample in data])

    def tukey_circles(self, data):
        num = []
        den = []
        crit = []
        radii = []
        xbar = []
        for sample in data:
            df = len(sample) - 1
            num.append(std(sample, ddof=1) ** 2 * df)
            den.append(df)
            crit.append(t.ppf(1 - self._alpha, df))
        mse = sum(num) / sum(den)
        for i, sample in enumerate(data):
            radii.append(fabs(crit[i]) * sqrt(mse / len(sample)))
            xbar.append(mean(sample))
        return tuple(zip(xbar, radii))

    def draw(self):
        """Draws the boxplots based on the set parameters."""

        # Setup the grid variables
        w_ratio = [1]
        if self._circles:
            w_ratio = [4, 1]
            self._ncols += 1
        if self._nqp:
            w_ratio.append(4 if self._circles else 1)
            self._ncols += 1
        groups, data = zip(*[
            (g, v['ind'].reset_index(drop=True)) for g, v in self._data.values.groupby('grp') if not v.empty]
        )

        # Create the quantile plot arrays
        prob = [probplot(v) for v in data]

        # Create the figure and gridspec
        if self._nqp and len(prob) > 0:
            self._xsize *= 2
        f = figure(figsize=(self._xsize, self._ysize))
        f.suptitle(self._title, fontsize=14)
        gs = GridSpec(self._nrows, self._ncols, width_ratios=w_ratio, wspace=0)

        # Draw the boxplots
        ax1 = subplot(gs[0])
        bp = ax1.boxplot(data, showmeans=True, labels=groups)
        setp(bp['boxes'], color='k')
        setp(bp['whiskers'], color='k')
        vp = ax1.violinplot(data, showextrema=False, showmedians=False, showmeans=False)
        for i in range(len(groups)):
            setp(vp['bodies'][i], facecolors=self.get_color(i))
        ax1.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
        if self._gmean:
            ax1.axhline(float(self.grand_mean(data)), c='k', linestyle='--', alpha=0.4)
        if self._gmedian:
            ax1.axhline(float(self.grand_median(data)), c='k', linestyle=':', alpha=0.4)
        if any([True if len(str(g)) > 9 else False for g in groups]) or len(groups) > 5:
            xticks(rotation=60)
        subplots_adjust(bottom=0.2)
        ylabel(self._yname)
        xlabel(self._xname)

        # Draw the Tukey-Kramer circles
        if self._circles:
            ax2 = subplot(gs[1], sharey=ax1)
            for i, (center, radius) in enumerate(self.tukey_circles(data)):
                c = Circle((0.5, center), radius=radius, facecolor='none', edgecolor=self.get_color(i))
                ax2.add_patch(c)
            # matplotlib 2.2.2 requires adjustable='datalim' to display properly.
            ax2.set_aspect('equal', adjustable='datalim')
            setp(ax2.get_xticklabels(), visible=False)
            setp(ax2.get_yticklabels(), visible=False)
            ax2.set_xticks([])

        # Draw the normal quantile plot
        if self._nqp and len(prob) > 0:
            ax3 = subplot(gs[2], sharey=ax1) if self._circles else subplot(gs[1], sharey=ax1)
            for i, g in enumerate(prob):
                osm = g[0][0]
                osr = g[0][1]
                slope = g[1][0]
                intercept = g[1][1]
                ax3.plot(osm, osr, marker='^', color=self.get_color(i), label=groups[i])
                ax3.plot(osm, slope * osm + intercept, linestyle='--', linewidth=2, color=self.get_color(i))
            ax3.xaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax3.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=0.75)
            ax3.legend(loc='best')
            xlabel("Quantiles")
            setp(ax3.get_yticklabels(), visible=False)

        # Save the figure to disk or display
        if self._save_to:
            savefig(self._save_to)
            close(f)
        else:
            show()
        pass
