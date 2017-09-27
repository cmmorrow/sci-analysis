"""sci_analysis package: analysis
Modules:
    analysis - sci_analysis test and calculation classes and functions
"""

from .hypo_tests import NormTest, KSTest, TwoSampleKSTest, MannWhitney, TTest, Anova, Kruskal, EqualVariance
from .comparison import LinearRegression, Correlation
from .stats import VectorStatistics, GroupStatistics, CategoricalStatistics


def determine_analysis_type(data):
    """Attempts to determine the type of data and returns the corresponding sci_analysis Data object.

    Parameters
    ----------
    data : array-like
        The sequence of unknown data type.

    Returns
    -------
    data : sci_analysis.data.Data
        A subclass of sci_analysis Data that corresponds to the analysis type to perform.
    """
    from numpy import (
        float16, float32, float64,
        int8, int16, int32, int64
    )
    from pandas import Series
    from ..data import is_iterable, is_vector, is_categorical, Vector, Categorical
    from .exc import NoDataError
    numeric_types = [float16, float32, float64,
                     int8, int16, int32, int64]
    if not is_iterable(data):
        raise ValueError('data cannot be a scalar value.')
    elif len(data) == 0:
        raise NoDataError
    elif is_vector(data):
        return data
    elif is_categorical(data):
        return data
    else:
        if not hasattr(data, 'dtype'):
            data = Series(data)
        if data.dtype in numeric_types:
            return Vector(data)
        else:
            return Categorical(data)


def analyse(xdata, ydata=None, groups=None, **kwargs):
    """
    Alias for analyze.

    Parameters
    ----------
    xdata : array-like
        The primary set of data.
    ydata : array-like
        The response data set.
    groups : array-like
        The group names used for a oneway analysis.
    kwargs

    Returns
    -------
    xdata, ydata : tuple(array-like, array-like)
        The input xdata and ydata.

    Notes
    -----
    xdata : array-like, ydata : None - Distribution
    xdata : array-like, ydata : array-like -- Bivariate
    xdata : list(array-like) or dict(array-like), ydata : None -- Oneway
    """
    return analyze(xdata, ydata=ydata, groups=groups, **kwargs)


def analyze(xdata, ydata=None, groups=None, **kwargs):
    """
    Automatically performs a statistical analysis based on the input arguments.

    Parameters
    ----------
    xdata : array-like
        The primary set of data.
    ydata : array-like
        The response data set.
    groups : array-like
        The group names used for a oneway analysis.

    Returns
    -------
    xdata, ydata : tuple(array-like, array-like)
        The input xdata and ydata.

    Notes
    -----
    xdata : array-like, ydata : None - Distribution
    xdata : array-like, ydata : array-like -- Bivariate
    xdata : list(array-like) or dict(array-like), ydata : None -- Oneway

    """
    from ..graphs import GraphHisto, GraphScatter, GraphBoxplot, GraphFrequency
    from ..data import (is_dict, is_iterable, is_group, is_dict_group, is_vector)
    from .exc import NoDataError
    groups = kwargs['groups'] if 'groups' in kwargs else None
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05
    debug = True if 'debug' in kwargs else False
    parms = kwargs
    tested = list()

    # if len(data) > 2:
    #     raise ValueError("analyze only accepts 2 arguments max. " + str(len(data)) + "arguments were passed.")
    if xdata is None:
        raise ValueError("xdata was not provided.")
    if not is_iterable(xdata):
        raise TypeError("xdata is not an array.")
    if len(xdata) == 0:
        raise NoDataError("No data was passed to analyze")

    # Compare Group Means and Variance
    if is_group(xdata) or is_dict_group(xdata):
        tested.append('Oneway')
        name = kwargs['name'] if 'name' in kwargs else 'Values'
        categories = kwargs['categories'] if 'categories' in kwargs else 'Categories'
        xname = kwargs['xname'] if 'xname' in kwargs else categories
        yname = kwargs['yname'] if 'yname' in kwargs else name
        parms['xname'] = xname
        parms['yname'] = yname
        if is_dict(xdata):
            groups = list(xdata.keys())
            xdata = list(xdata.values())
        parms['groups'] = groups

        # Show the box plot and stats
        GraphBoxplot(*xdata, **parms)
        GroupStatistics(*xdata, groups=groups)

        if len(xdata) == 2:
            norm = NormTest(*xdata, alpha=alpha, display=False)
            if norm.p_value > alpha:
                TTest(xdata[0], xdata[1])
                tested.append('TTest')
            elif len(xdata[0]) > 25 and len(xdata[1]) > 25:
                MannWhitney(xdata[0], xdata[1])
                tested.append('MannWhitney')
            else:
                TwoSampleKSTest(xdata[0], xdata[1])
                tested.append('TwoSampleKSTest')
        else:
            e = EqualVariance(*xdata, alpha=alpha)

            # If normally distributed and variances are equal, perform one-way ANOVA
            # Otherwise, perform a non-parametric Kruskal-Wallis test
            if e.test_type == 'Bartlett' and e.p_value > alpha:
                    Anova(*xdata, alpha=alpha)
                    tested.append('Anova')
            else:
                    Kruskal(*xdata, alpha=alpha)
                    tested.append('Kruskal')
        return tested if debug else None

    # Correlation and Linear Regression
    elif is_iterable(xdata) and is_iterable(ydata):
        tested.append('Bivariate')
        xname = kwargs['xname'] if 'xname' in kwargs else 'Predictor'
        yname = kwargs['yname'] if 'yname' in kwargs else 'Response'
        parms['xname'] = xname
        parms['yname'] = yname

        # Show the scatter plot, correlation and regression stats
        GraphScatter(xdata, ydata, **parms)
        LinearRegression(xdata, ydata, alpha=alpha)
        Correlation(xdata, ydata, alpha=alpha)
        return tested if debug else None

    # Histogram and Basic Stats or Categories and Frequencies
    elif is_iterable(xdata):
        xdata = determine_analysis_type(xdata)
        if is_vector(xdata):
            tested.append('Distribution')

            # Show the histogram and stats
            out_stats = VectorStatistics(xdata, display=False)
            if 'distribution' in kwargs:
                distro = kwargs['distribution']
                distro_class = getattr(__import__('scipy.stats',
                                                  globals(),
                                                  locals(),
                                                  [distro], 0), distro)
                parms = distro_class.fit(xdata)
                fit = KSTest(xdata, distribution=distro, parms=parms, alpha=alpha, display=False)
                tested.append('KSTest')
            else:
                fit = NormTest(xdata, alpha=alpha, display=False)
                tested.append('NormTest')
            GraphHisto(xdata,
                       mean="{: .4f}".format(out_stats.mean),
                       std_dev="{: .4f}".format(out_stats.std_dev),
                       **kwargs)
            print(out_stats)
            print(fit)
            return tested if debug else None
        else:
            tested.append('Frequencies')

            # Show the histogram and stats
            GraphFrequency(xdata, **kwargs)
            CategoricalStatistics(xdata, **kwargs)
            return tested if debug else None
    else:
        return xdata, ydata
