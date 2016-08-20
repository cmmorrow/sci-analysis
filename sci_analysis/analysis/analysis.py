"""Module: analysis.py
Classes:
    Analysis - Generic analysis root class.
    Test - Generic statistical test class.
    GroupTest - Perform a test on multiple vectors that are passed as a tuple of arbitrary length.
    Comparison - Perform a test on two independent vectors of equal length.
    NormTest - Tests for whether data is normally distributed or not.
    GroupNormTest - Tests a group of data to see if they are normally distributed or not.
    TTest - Performs a T-Test on the two provided vectors.
    LinearRegression - Performs a linear regression between two vectors.
    Correlation - Performs a pearson or spearman correlation between two vectors.
    Anova - Performs a one-way ANOVA on a group of vectors.
    Kruskal - Performs a non-parametric Kruskal-Wallis test on a group of vectors.
    EqualVariance - Checks a group of vectors for equal variance.
    VectorStatistics - Reports basic summary stats for a provided vector.
    GroupStatistics - Reports basic summary stats for a group of vectors.
Functions:
    analyze - Magic method for performing quick data analysis.
"""
# Python3 compatability
from __future__ import absolute_import
from __future__ import print_function

# Scipy imports
from scipy.stats import linregress, shapiro, pearsonr, spearmanr, ttest_ind, \
    ttest_1samp, f_oneway, kruskal, bartlett, levene, skew, kurtosis, kstest, sem, ks_2samp, mannwhitneyu

# Numpy imports
from numpy import mean, std, median, amin, amax, percentile

# Local imports
from ..operations.data_operations import is_dict, is_iterable, is_group, is_dict_group
from ..graphs.graph import GraphHisto, GraphScatter, GraphBoxplot
from ..data.data import assign


class MinimumSizeError(Exception):
    pass


class NoDataError(Exception):
    pass


class Analysis(object):
    """Generic analysis root class.

    Members:
        _data - the data used for analysis.
        _display - flag for whether to display the analysis output.
        _results - A dict of the results of the test.

    Methods:
        logic - This method needs to run the analysis, set the results member, and display the output at bare minimum.
        run - This method should return the results of the specific analysis.
        output - This method shouldn't return a value and only produce a side-effect.
    """

    _name = "Analysis"

    def __init__(self, data, display=True):
        """Initialize the data and results members.

        Override this method to initialize additional members or perform
        checks on data.
        """
        self._data = data
        self._display = display
        self._results = {}

    @property
    def name(self):
        """The name of the test class"""
        return self._name

    @property
    def data(self):
        """The data used for analysis"""
        return self._data

    @property
    def results(self):
        """A dict of the results returned by the run method"""
        return self._results

    def logic(self):
        """This method needs to run the analysis, set the results member, and
        display the output at bare minimum.

        Override this method to modify the execution sequence of the analysis.
        """
        if self._data is None:
            pass
        self.run()
        if self._display:
            print(self)

    def run(self):
        """This method should perform the specific analysis and set the results dict.

        Override this method to perform a specific analysis or calculation.
        """
        pass

    def output(self, name, order=list(), no_format=list(), precision=4):
        """Print the results of the test in a user-friendly format"""
        label_max_length = 0

        def format_output(n, v):
            """Format the results with a consistent look"""
            return '{:{}s}'.format(n, label_max_length) + " = " + '{:< .{}f}'.format(v, precision)

        def no_precision_output(n, v):
            return '{:{}s}'.format(n, label_max_length) + " = " + " " + str(v)

        for label in self._results.keys():
            if len(label) > label_max_length:
                label_max_length = len(label)

        report = [
            ' ',
            name,
            '-' * len(name),
            ' ',
        ]

        if order:
            for key in order:
                for label, value in self._results.items():
                    if label == key:
                        if label in no_format:
                            report.append(no_precision_output(label, value))
                        else:
                            report.append(format_output(label, value))
                        continue
        else:
            for label, value in self._results.items():
                report.append(format_output(label, value))

        report.append(" ")

        return "\n".join(report)

    def __str__(self):
        return self.output(self._name)


class GroupAnalysis(Analysis):

    def output(self, name, order=list(), no_format=list(), precision=4, spacing=14):
        grid = list()

        def format_header(names):
            line = ""
            for n in names:
                line += '{:{}s}'.format(n, spacing)
            return line

        def format_row(_row, _order):
            line = ""
            for _label in _order:
                for k, v in _row.items():
                    if k == _label:
                        if k in no_format:
                            line += '{:<{}s}'.format(str(v), spacing)
                        else:
                            line += '{:< {}.{}f}'.format(v, spacing, precision)
                        continue
            return line

        header = format_header(order)
        for row in self._results:
            grid.append(format_row(row, order))

        return '\n'.join((
            name,
            " ",
            header,
            "-" * len(header),
            '\n'.join(grid),
            " "
        ))


class Test(Analysis):
    """Generic statistical test class.
    Members:
        _name - The name of the test.
        _h0 - Prints the null hypothesis.
        _ha - Prints the alternate hypothesis.
        _data - the data used for analysis.
        _display - flag for whether to display the analysis output.
        _alpha - the statistical significance of the test.
        _results - A dict of the results of the test.
    Methods:
        logic - If the result is greater than the significance, print the null hypothesis, otherwise,
            the alternate hypothesis.
        run - This method should return the results of the specific analysis.
        output - This method shouldn't return a value and only produce a side-effect.
    """

    _name = "Test"
    _statistic_name = "test"
    _h0 = "H0: "
    _ha = "HA: "
    _min_size = 2

    def __init__(self, *args, **kwargs):
        """Initialize the object"""

        self._alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05
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

        # set the _data and _display members
        super(Test, self).__init__(data, display=(kwargs['display'] if 'display' in kwargs else True))

        # Run the test and display the results
        self.logic()

    @property
    def statistic(self):
        """The test statistic returned by the function called in the run method"""
        return self._results['statistic']

    @property
    def p_value(self):
        """The p-value returned by the function called in the run method"""
        return self._results['p value']

    def output(self, name, order=list(), no_format=list(), precision=4):
        """Print the results of the test in a user-friendly format"""
        label_max_length = 0

        def format_output(n, v):
            """Format the results with a consistent look"""
            return '{:{}s}'.format(n, label_max_length) + " = " + '{:< .{}f}'.format(v, precision)

        def no_precision_output(n, v):
            return '{:{}s}'.format(n, label_max_length) + " = " + " " + str(v)

        for label in self._results.keys():
            if len(label) > label_max_length:
                label_max_length = len(label)

        report = [
            ' ',
            name,
            '-' * len(name),
            ' ',
        ]

        if order:
            for key in order:
                for label, value in self._results.items():
                    if label == key:
                        if label in no_format:
                            report.append(no_precision_output(label, value))
                        else:
                            report.append(format_output(label, value))
                        continue
        else:
            for label, value in self._results.items():
                report.append(format_output(label, value))

        report.append(" ")
        report.append(self._h0 if self.p_value > self._alpha else self._ha)
        report.append(" ")

        return "\n".join(report)


class Comparison(Analysis):
    """Perform a test on two independent vectors of equal length."""

    _min_size = 3
    _name = "Comparison"
    _h0 = "H0: "
    _ha = "HA: "

    def __init__(self, xdata, ydata, alpha=0.05, display=True):
        self._alpha = alpha
        x, y = assign(xdata, ydata)
        if x is None or y is None:
            raise NoDataError("Cannot perform test because there is no data")
        try:
            x, y = x.data_prep(y)
        except TypeError:
            raise NoDataError("Cannot perform test because there is no data")
        if len(x) <= self._min_size or len(y) <= self._min_size:
            raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
        super(Comparison, self).__init__([x, y], display=display)
        self.logic()

    @property
    def xdata(self):
        """The predictor vector for comparison tests"""
        return self.data[0]

    @property
    def ydata(self):
        """The response vector for comparison tests"""
        return self.data[1]

    @property
    def predictor(self):
        """The predictor vector for comparison tests"""
        return self.data[0]

    @property
    def response(self):
        """The response vector for comparison tests"""
        return self.data[1]

    @property
    def statistic(self):
        """The test statistic returned by the function called in the run method"""
        return self._results['statistic']

    @property
    def p_value(self):
        """The p-value returned by the function called in the run method"""
        return self._results['p value']

    def output(self, name, order=list(), no_format=list(), precision=4):
        """Print the results of the test in a user-friendly format"""
        label_max_length = 0

        def format_output(n, v):
            """Format the results with a consistent look"""
            return '{:{}s}'.format(n, label_max_length) + " = " + '{:< .{}f}'.format(v, precision)

        def no_precision_output(n, v):
            return '{:{}s}'.format(n, label_max_length) + " = " + " " + str(v)

        for label in self._results.keys():
            if len(label) > label_max_length:
                label_max_length = len(label)

        report = [
            ' ',
            name,
            '-' * len(name),
            ' ',
        ]

        if order:
            for key in order:
                for label, value in self._results.items():
                    if label == key:
                        if label in no_format:
                            report.append(no_precision_output(label, value))
                        else:
                            report.append(format_output(label, value))
                        continue
        else:
            for label, value in self._results.items():
                report.append(format_output(label, value))

        if self.p_value:
            report.append(" ")
            report.append(self._h0 if self.p_value > self._alpha else self._ha)
            report.append(" ")

        return "\n".join(report)


class NormTest(Test):
    """Tests for whether data is normally distributed or not."""

    _name = "Shapiro-Wilk test for normality"
    _h0 = "H0: Data is normally distributed"
    _ha = "HA: Data is not normally distributed"

    def run(self):
        if not is_group(self._data):
            w_value, p_value = shapiro(self.data)
        else:
            w_value = list()
            p_value = list()
            for d in self._data:
                _w, _p = shapiro(d)
                w_value.append(_w)
                p_value.append(_p)
            min_p = min(p_value)
            w_value = w_value[p_value.index(min_p)]
            p_value = min_p
        self._results.update({'W value': w_value, 'p value': p_value})

    @property
    def statistic(self):
        return self._results['W value']

    @property
    def w_value(self):
        return self._results['W value']


class KSTest(Test):
    """Tests whether data comes from a specified distribution or not."""

    _name = "Kolmogorov-Smirnov Test"

    def __init__(self, data, distribution='norm', parms=(), alpha=0.05, display=True):
        self._distribution = distribution
        self._parms = parms
        self._h0 = "H0: Data is matched to the " + self.distribution + " distribution"
        self._ha = "HA: Data is not from the " + self.distribution + " distribution"
        super(KSTest, self).__init__(data, alpha=alpha, display=display)

    def run(self):
        args = [self._data, self._distribution]
        if self._parms:
            args.append(self._parms)
        d_value, p_value = kstest(*args)
        self._results.update({'D value': d_value, 'p value': p_value})

    @property
    def distribution(self):
        """Return the distribution that data is being compared against"""
        return self._distribution

    @property
    def statistic(self):
        return self._results['D value']

    @property
    def d_value(self):
        return self._results['D value']


class TwoSampleKSTest(Test):
    """Tests whether two independent vectors come from the same distribution"""

    _name = "Two Sample Kolmogorov-Smirnov Test"
    _h0 = "H0: Both samples come from the same distribution"
    _ha = "HA: Samples do not come from the same distribution"

    def __init__(self, a, b, alpha=0.05, display=True):
        super(TwoSampleKSTest, self).__init__(a, b, alpha=alpha, display=display)

    def run(self):
        d_value, p_value = ks_2samp(*self._data)
        self._results.update({'D value': d_value, 'p value': p_value})

    @property
    def statistic(self):
        return self._results['D value']

    @property
    def d_value(self):
        return self._results['D value']


class MannWhitney(Test):
    """Performs a Mann Whitney U Test on two vectors"""

    _name = "Mann Whitney U Test"
    _h0 = "H0: Locations are matched"
    _ha = "HA: Locations are not matched"
    _min_size = 30

    def run(self):
        u_value, p_value = mannwhitneyu(*self._data)
        self._results.update({'u value': u_value, 'p value': p_value * 2})

    @property
    def statistic(self):
        return self._results['u value']

    @property
    def u_value(self):
        return self._results['u value']


class TTest(Test):
    """Performs a T-Test on the two provided vectors."""

    _name = {'1_sample': '1 Sample T Test', 't_test': 'T Test', 'welch_t': "Welch's T Test"}
    _h0 = "H0: Means are matched"
    _ha = "HA: Means are significantly different"
    _min_size = 3

    def __init__(self, xdata, ydata, alpha=0.05, display=True):
        self._mu = None
        self._test = None
        if not is_iterable(ydata):
            self._mu = float(ydata)
            super(TTest, self).__init__(xdata, alpha=alpha, display=display)
        else:
            super(TTest, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        if self._mu:
            t, p = ttest_1samp(self._data, self._mu, axis=0)
            test = "1_sample"
        else:
            if not is_group(self._data):
                raise NoDataError("Cannot perform the test because there is no data")
            if EqualVariance(*self._data, display=False, alpha=self._alpha).p_value > self._alpha:
                t, p = ttest_ind(*self._data, equal_var=True, axis=0)
                test = 't_test'
            else:
                t, p = ttest_ind(*self._data, equal_var=False, axis=0)
                test = 'welch_t'
        self._test = test
        self._results.update({'p value': p, 't value': float(t)})

    @property
    def test_type(self):
        return self._test

    @property
    def mu(self):
        return self._mu

    @property
    def t_value(self):
        return self._results['t value']

    @property
    def statistic(self):
        return self._results['t value']

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        return self.output(self._name[self._test])


class LinearRegression(Comparison):
    """Performs a linear regression between two vectors."""

    _name = "Linear Regression"
    _h0 = "H0: There is no significant relationship between predictor and response"
    _ha = "HA: There is a significant relationship between predictor and response"

    def __init__(self, xdata, ydata, alpha=0.05, display=True):
        super(LinearRegression, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        slope, intercept, r2, p_value, std_err = linregress(self.xdata, self.ydata)
        count = len(self.xdata)
        self._results.update({'Count': count,
                              'Slope': slope,
                              'Intercept': intercept,
                              'R^2': r2,
                              'Std Err': std_err,
                              'p value': p_value})

    @property
    def slope(self):
        return self._results['Slope']

    @property
    def intercept(self):
        return self._results['Intercept']

    @property
    def r_squared(self):
        return self._results['R^2']

    @property
    def statistic(self):
        return self._results['R^2']

    @property
    def std_err(self):
        return self._results['Std Err']

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        return self.output(self._name, order=['Count', 'Slope', 'Intercept', 'R^2', 'Std Err', 'p value'],
                           no_format=['Count'])


class Correlation(Comparison):
    """Performs a pearson or spearman correlation between two vectors."""

    _name = {'pearson': 'Pearson Correlation Coefficient', 'spearman': 'Spearman Correlation Coefficient'}
    _h0 = "H0: There is no significant relationship between predictor and response"
    _ha = "HA: There is a significant relationship between predictor and response"

    def __init__(self, xdata, ydata, alpha=0.05, display=True):
        self._test = None
        super(Correlation, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        if NormTest(self.xdata, self.ydata, display=False, alpha=self._alpha).p_value > self._alpha:
            r_value, p_value = pearsonr(self.xdata, self.ydata)
            r = "pearson"
        else:
            r_value, p_value = spearmanr(self.xdata, self.ydata)
            r = "spearman"
        self._test = r
        self._results.update({'r value': r_value, 'p value': p_value})

    @property
    def r_value(self):
        """The correlation coefficient returned by the the determined test type"""
        return self._results['r value']

    @property
    def statistic(self):
        return self._results['r value']

    @property
    def test_type(self):
        """The test that was used to determine the correlation coefficient"""
        return self._test

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        return self.output(self._name[self._test])


class Anova(Test):
    """Performs a one-way ANOVA on a group of vectors."""

    _name = "Oneway ANOVA"
    _h0 = "H0: Group means are matched"
    _ha = "HA: Group means are not matched"

    def run(self):
        f_value, p_value = f_oneway(*self.data)
        self._results.update({'p value': p_value, 'f value': f_value})

    @property
    def f_value(self):
        """The f value returned by the ANOVA f test"""
        return self._results['f value']

    @property
    def statistic(self):
        return self._results['f value']


class Kruskal(Test):
    """Performs a non-parametric Kruskal-Wallis test on a group of vectors."""

    _name = "Kruskal-Wallis"
    _h0 = "H0: Group means are matched"
    _ha = "HA: Group means are not matched"

    def run(self):
        h_value, p_value = kruskal(*self.data)
        self._results.update({'p value': p_value, 'h value': h_value})

    @property
    def h_value(self):
        """The h value returned by the Kruskal test"""
        return self._results['h value']

    @property
    def statistic(self):
        return self._results['h value']


class EqualVariance(Test):
    """Checks a group of vectors for equal variance."""

    _name = {'Bartlett': 'Bartlett Test', 'Levene': 'Levene Test'}
    _h0 = "H0: Variances are equal"
    _ha = "HA: Variances are not equal"

    def __init__(self, *data, **kwargs):
        self._test = None
        super(EqualVariance, self).__init__(*data, **kwargs)

    def run(self):
        if len(self._data) < self._min_size:
            pass
        if NormTest(*self._data, display=False, alpha=self._alpha).p_value > self._alpha:
            statistic, p_value = bartlett(*self._data)
            self._test = 'Bartlett'
            self._results.update({'p value': p_value, 'T value': statistic})
        else:
            statistic, p_value = levene(*self._data)
            self._test = 'Levene'
            self._results.update({'p value': p_value, 'W value': statistic})

    @property
    def t_value(self):
        return self._results['T value']

    @property
    def w_value(self):
        return self._results['W value']

    @property
    def statistic(self):
        try:
            s = self._results['W value']
        except KeyError:
            s = self._results['T value']
        return s

    @property
    def test_type(self):
        """The test that was used to check for equal variance"""
        return self._test

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        return self.output(self._name[self._test])


class VectorStatistics(Analysis):
    """Reports basic summary stats for a provided vector."""

    _min_size = 1
    _name = 'Statistics'

    def __init__(self, data, sample=True, display=True):
        self._sample = sample
        d = assign(data).data_prep()
        if d is None:
            raise NoDataError("Cannot perform the test because there is no data")
        if len(d) <= self._min_size:
            raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))

        super(VectorStatistics, self).__init__(d, display=display)
        self.logic()

    def run(self):
        dof = 0
        if self._sample:
            dof = 1
        count = len(self._data)
        avg = mean(self._data)
        sd = std(self._data, ddof=dof)
        error = sem(self._data, 0, dof)
        med = median(self._data)
        vmin = amin(self._data)
        vmax = amax(self._data)
        vrange = vmax - vmin
        sk = skew(self._data)
        kurt = kurtosis(self._data)
        q1 = percentile(self._data, 25)
        q3 = percentile(self._data, 75)
        iqr = q3 - q1
        self._results = {"Count": count,
                         "Mean": avg,
                         "Std Dev": sd,
                         "Std Error": error,
                         "50%": med,
                         "Minimum": vmin,
                         "Maximum": vmax,
                         "Range": vrange,
                         "Skewness": sk,
                         "Kurtosis": kurt,
                         "25%": q1,
                         "75%": q3,
                         "IQR": iqr}

    @property
    def count(self):
        return self._results['Count']

    @property
    def mean(self):
        return self._results['Mean']

    @property
    def std_dev(self):
        return self._results['Std Dev']

    @property
    def std_err(self):
        return self._results['Std Error']

    @property
    def median(self):
        return self._results['50%']

    @property
    def minimum(self):
        return self._results['Minimum']

    @property
    def maximum(self):
        return self._results['Maximum']

    @property
    def range(self):
        return self._results['Range']

    @property
    def skewness(self):
        return self._results['Skewness']

    @property
    def kurtosis(self):
        return self._results['Kurtosis']

    @property
    def q1(self):
        return self._results['25%']

    @property
    def q3(self):
        return self._results['75%']

    @property
    def iqr(self):
        return self._results['IQR']

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        return self.output(self._name, order=['Count',
                                              'Mean',
                                              'Std Dev',
                                              'Std Error',
                                              'Skewness',
                                              'Kurtosis',
                                              'Maximum',
                                              '75%',
                                              '50%',
                                              '25%',
                                              'Minimum',
                                              'IQR',
                                              'Range'], no_format=['Count'])


class GroupStatistics(GroupAnalysis):
    """Reports basic summary stats for a group of vectors."""

    _min_size = 1
    _name = 'Group Statistics'

    def __init__(self, *args, **kwargs):
        groups = kwargs['groups'] if 'groups' in kwargs else None
        display = kwargs['display'] if 'display' in kwargs else True
        if not is_dict(args[0]):
            _data = dict(zip(groups, args)) if groups else dict(zip(list(range(1, len(args) + 1)), args))
        else:
            _data = args[0]
        data = dict()
        for g, d in _data.items():
            clean = assign(d).data_prep()
            if clean is None:
                continue
            if len(clean) <= self._min_size:
                raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
            data.update({g: clean})
        if len(data) < 1:
            raise NoDataError("Cannot perform test because there is no data")
        if len(data) == 1:
            data = data[0]
        super(GroupStatistics, self).__init__(data, display=display)
        self.logic()

    def logic(self):
        if not self._data:
            pass
        self._results = list()
        self.run()
        if self._display:
            print(self)

    def run(self):
        for group, vector in self._data.items():
            count = len(vector)
            avg = mean(vector)
            sd = std(vector, ddof=1)
            vmax = amax(vector)
            vmin = amin(vector)
            q2 = median(vector)
            row_result = {"Group": group,
                          "Count": count,
                          "Mean": avg,
                          "Std Dev": sd,
                          "Max": vmax,
                          "Median": q2,
                          "Min": vmin}
            self._results.append(row_result)

    def __str__(self):
        return self.output(self._name, ['Count', 'Mean', 'Std Dev', 'Min', 'Median', 'Max', 'Group'],
                           no_format=['Count', 'Group'])


def analyze(*data, **kwargs):
    """Magic method for performing quick data analysis.

    :param xdata: A Vector, numPy Array or sequence like object
    :param ydata: An optional secondary Vector, numPy Array or sequence object
    :param groups: A list of group names. The box plots will be graphed in order of groups
    :param name: The response variable label
    :param xname: The predictor variable (x-axis) label
    :param yname: The response variable (y-axis) label
    :param alpha: The significance level of the test
    :param categories: The x-axis label when performing a group analysis
    :return: A tuple of xdata and ydata
    """
    groups = kwargs['groups'] if 'groups' in kwargs else None
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05
    debug = True if 'debug' in kwargs else False
    xdata = data[0]
    ydata = data[1] if len(data) > 1 else None
    parms = kwargs
    tested = list()

    if len(data) > 2:
        raise ValueError("analyze only accepts 2 arguments max. " + str(len(data)) + "arguments were passed.")
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

    # Histogram and Basic Stats
    elif is_iterable(xdata):
        tested.append('Distribution')

        # Show the histogram and stats
        stats = VectorStatistics(xdata, display=False)
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
        GraphHisto(xdata, mean="{: .4f}".format(stats.mean), std_dev="{: .4f}".format(stats.std_dev), **kwargs)
        print(stats)
        print(fit)
        return tested if debug else None
    else:
        return xdata, ydata
