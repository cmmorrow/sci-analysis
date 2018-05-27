"""sci_analysis package: analysis
Modules:
    analysis - sci_analysis test and calculation classes and functions
"""

from .hypo_tests import NormTest, KSTest, TwoSampleKSTest, MannWhitney, TTest, Anova, Kruskal, EqualVariance
from .comparison import LinearRegression, Correlation, GroupCorrelation, GroupLinearRegression
from .stats import VectorStatistics, GroupStatistics, GroupStatisticsStacked, CategoricalStatistics


def determine_analysis_type(data, other=None, groups=None):
    """Attempts to determine the type of data and returns the corresponding sci_analysis Data object.

    Parameters
    ----------
    data : array-like
        The sequence of unknown data type.
    other : array-like or None
        A second sequence of unknown data type.
    groups : array-like or None
        The group names to include if data is determined to be a Vector.

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
        if other is not None:
            if not hasattr(other, 'dtype'):
                other = Series(other)
        if data.dtype in numeric_types:
            if other is not None and other.dtype in numeric_types:
                if groups is not None:
                    return Vector(data, other=other, groups=groups)
                else:
                    return Vector(data, other=other)
            else:
                if groups is not None:
                    return Vector(data, groups=groups)
                else:
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
        The response or secondary set of data.
    groups : array-like
        The group names used for location testing or Bivariate analysis.
    alpha : float
        The sensitivity to use for hypothesis tests.

    Returns
    -------
    xdata, ydata : tuple(array-like, array-like)
        The input xdata and ydata.

    Notes
    -----
    xdata : array-like(num), ydata : None --- Distribution
    xdata : array-like(str), ydata : None --- Frequencies
    xdata : array-like(num), ydata : array-like(num) --- Bivariate
    xdata : array-like(num), ydata : array-like(num), groups : array-like --- Group Bivariate
    xdata : list(array-like(num)), ydata : None --- Location Test(unstacked)
    xdata : list(array-like(num)), ydata : None, groups : array-like --- Location Test(unstacked)
    xdata : dict(array-like(num)), ydata : None --- Location Test(unstacked)
    xdata : array-like(num), ydata : None, groups : array-like --- Location Test(stacked)
    """
    return analyze(xdata, ydata=ydata, groups=groups, **kwargs)


def analyze(xdata, ydata=None, groups=None, alpha=0.05, **kwargs):
    """
    Automatically performs a statistical analysis based on the input arguments.

    Parameters
    ----------
    xdata : array-like
        The primary set of data.
    ydata : array-like
        The response or secondary set of data.
    groups : array-like
        The group names used for location testing or Bivariate analysis.
    alpha : float
        The sensitivity to use for hypothesis tests.

    Returns
    -------
    xdata, ydata : tuple(array-like, array-like)
        The input xdata and ydata.

    Notes
    -----
    xdata : array-like(num), ydata : None --- Distribution
    xdata : array-like(str), ydata : None --- Frequencies
    xdata : array-like(num), ydata : array-like(num) --- Bivariate
    xdata : array-like(num), ydata : array-like(num), groups : array-like --- Group Bivariate
    xdata : list(array-like(num)), ydata : None --- Location Test(unstacked)
    xdata : list(array-like(num)), ydata : None, groups : array-like --- Location Test(unstacked)
    xdata : dict(array-like(num)), ydata : None --- Location Test(unstacked)
    xdata : array-like(num), ydata : None, groups : array-like --- Location Test(stacked)
    """
    from ..graphs import GraphHisto, GraphScatter, GraphBoxplot, GraphFrequency, GraphGroupScatter
    from ..data import (is_dict, is_iterable, is_group, is_dict_group, is_vector)
    from .exc import NoDataError
    debug = True if 'debug' in kwargs else False
    tested = list()

    if xdata is None:
        raise ValueError("xdata was not provided.")
    if not is_iterable(xdata):
        raise TypeError("xdata is not an array.")
    if len(xdata) == 0:
        raise NoDataError("No data was passed to analyze")

    # Compare Group Means and Variance
    if is_group(xdata) or is_dict_group(xdata):
        tested.append('Oneway')
        if is_dict(xdata):
            if groups is not None:
                GraphBoxplot(xdata, groups=groups, **kwargs)
            else:
                GraphBoxplot(xdata, **kwargs)
            groups = list(xdata.keys())
            xdata = list(xdata.values())
        else:
            if groups is not None:
                GraphBoxplot(*xdata, groups=groups, **kwargs)
            else:
                GraphBoxplot(*xdata, **kwargs)
        out_stats = GroupStatistics(*xdata, groups=groups, display=False)
        # Show the box plot and stats
        print(out_stats)

        if len(xdata) == 2:
            norm = NormTest(*xdata, alpha=alpha, display=False)
            if norm.p_value > alpha:
                TTest(xdata[0], xdata[1], alpha=alpha)
                tested.append('TTest')
            elif len(xdata[0]) > 25 and len(xdata[1]) > 25:
                MannWhitney(xdata[0], xdata[1], alpha=alpha)
                tested.append('MannWhitney')
            else:
                TwoSampleKSTest(xdata[0], xdata[1], alpha=alpha)
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

    if ydata is not None:
        _data = determine_analysis_type(xdata, other=ydata, groups=groups)
    else:
        _data = determine_analysis_type(xdata, groups=groups)

    if is_vector(_data) and not _data.other.empty:
        # Correlation and Linear Regression
        if len(_data.groups) > 1:
            tested.append('Group Bivariate')

            # Show the scatter plot, correlation and regression stats
            GraphGroupScatter(_data, **kwargs)
            GroupLinearRegression(_data, alpha=alpha)
            GroupCorrelation(_data, alpha=alpha)
            return tested if debug else None
        else:
            tested.append('Bivariate')

            # Show the scatter plot, correlation and regression stats
            GraphScatter(_data, **kwargs)
            LinearRegression(_data, alpha=alpha)
            Correlation(_data, alpha=alpha)
            return tested if debug else None
    elif is_vector(_data) and len(_data.groups) > 1:
        # Compare Stacked Group Means and Variance
        tested.append('Stacked Oneway')

        # Show the box plot and stats
        out_stats = GroupStatisticsStacked(_data, display=False)
        GraphBoxplot(_data, gmean=out_stats.gmean, gmedian=out_stats.gmedian, **kwargs)
        print(out_stats)

        group_data = tuple(_data.groups.values())
        if len(group_data) == 2:
            norm = NormTest(*group_data, alpha=alpha, display=False)
            if norm.p_value > alpha:
                TTest(*group_data)
                tested.append('TTest')
            elif len(group_data[0]) > 25 and len(group_data[1]) > 25:
                MannWhitney(*group_data)
                tested.append('MannWhitney')
            else:
                TwoSampleKSTest(*group_data)
                tested.append('TwoSampleKSTest')
        else:
            e = EqualVariance(*group_data, alpha=alpha)
            if e.test_type == 'Bartlett' and e.p_value > alpha:
                Anova(*group_data, alpha=alpha)
                tested.append('Anova')
            else:
                Kruskal(*group_data, alpha=alpha)
                tested.append('Kruskal')
        return tested if debug else None
    else:
        # Histogram and Basic Stats or Categories and Frequencies
        if is_vector(_data):
            tested.append('Distribution')

            # Show the histogram and stats
            if 'sample' in kwargs:
                out_stats = VectorStatistics(_data, sample=kwargs['sample'], display=False)
            else:
                out_stats = VectorStatistics(_data, display=False)
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
            GraphHisto(_data, mean=out_stats.mean, std_dev=out_stats.std_dev, **kwargs)
            print(out_stats)
            print(fit)
            return tested if debug else None
        else:
            tested.append('Frequencies')

            # Show the histogram and stats
            GraphFrequency(_data, **kwargs)
            CategoricalStatistics(xdata, **kwargs)
            return tested if debug else None
