# Scipy imports
from scipy.stats import linregress, pearsonr, spearmanr
from pandas import DataFrame

from ..data import Vector, is_vector
from .base import Analysis, std_output
from .exc import NoDataError, MinimumSizeError
from .hypo_tests import NormTest


class Comparison(Analysis):
    """Perform a test on two independent vectors of equal length."""

    _min_size = 3
    _name = "Comparison"
    _h0 = "H0: "
    _ha = "HA: "
    _default_alpha = 0.05

    def __init__(self, xdata, ydata=None, alpha=None, display=True):
        self._alpha = alpha or self._default_alpha
        if ydata is None:
            if is_vector(xdata):
                v = xdata
            else:
                raise AttributeError('ydata argument cannot be None.')
        else:
            v = Vector(xdata, other=ydata)
        if v.data.empty or v.other.empty:
            raise NoDataError("Cannot perform test because there is no data")
        if len(v.data) <= self._min_size or len(v.other) <= self._min_size:
            raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
        super(Comparison, self).__init__(v, display=display)
        self.logic()

    @property
    def xdata(self):
        """The predictor vector for comparison tests"""
        return self.data.data

    @property
    def ydata(self):
        """The response vector for comparison tests"""
        return self.data.other

    @property
    def predictor(self):
        """The predictor vector for comparison tests"""
        return self.data.data

    @property
    def response(self):
        """The response vector for comparison tests"""
        return self.data.other

    @property
    def statistic(self):
        """The test statistic returned by the function called in the run method"""
        # TODO: Need to catch the case where self._results is an empty dictionary.
        return self._results['statistic']

    @property
    def p_value(self):
        """The p-value returned by the function called in the run method"""
        return self._results['p value']

    def __str__(self):
        out = list()
        order = list()
        res = list(self._results.keys())
        if 'p value' in res:
            order.append('p value')
            res.remove('p value')
        order.extend(res)

        out.append(std_output(self.name, self._results, reversed(order)))
        out.append('')
        out.append(self._h0 if self.p_value > self._alpha else self._ha)
        out.append('')
        return '\n'.join(out)

    def run(self):
        raise NotImplementedError


class LinearRegression(Comparison):
    """Performs a linear regression between two vectors."""

    _name = "Linear Regression"
    _n = 'n'
    _slope = 'Slope'
    _intercept = 'Intercept'
    _r_value = 'r'
    _r_squared = 'r^2'
    _std_err = 'Std Err'
    _p_value = 'p value'

    def __init__(self, xdata, ydata=None, alpha=None, display=True):
        super(LinearRegression, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        slope, intercept, r, p_value, std_err = linregress(self.xdata, self.ydata)
        count = len(self.xdata)
        self._results.update({
            self._n: count,
            self._slope: slope,
            self._intercept: intercept,
            self._r_value: r,
            self._r_squared: r ** 2,
            self._std_err: std_err,
            self._p_value: p_value
        })

    @property
    def slope(self):
        return self._results[self._slope]

    @property
    def intercept(self):
        return self._results[self._intercept]

    @property
    def r_squared(self):
        return self._results[self._r_squared]

    @property
    def r_value(self):
        return self._results[self._r_value]

    @property
    def statistic(self):
        return self._results[self._r_squared]

    @property
    def std_err(self):
        return self._results[self._std_err]

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        out = list()
        order = [
            self._n,
            self._slope,
            self._intercept,
            self._r_value,
            self._r_squared,
            self._std_err,
            self._p_value
        ]
        out.append(std_output(self._name, self._results, order=order))
        out.append('')
        return '\n'.join(out)


class Correlation(Comparison):
    """Performs a pearson or spearman correlation between two vectors."""

    _names = {'pearson': 'Pearson Correlation Coefficient', 'spearman': 'Spearman Correlation Coefficient'}
    _h0 = "H0: There is no significant relationship between predictor and response"
    _ha = "HA: There is a significant relationship between predictor and response"
    _r_value = 'r value'
    _p_value = 'p value'
    _alpha_name = 'alpha'

    def __init__(self, xdata, ydata=None, alpha=None, display=True):
        self._test = None
        super(Correlation, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        if NormTest(self.xdata, self.ydata, display=False, alpha=self._alpha).p_value > self._alpha:
            r_value, p_value = pearsonr(self.xdata, self.ydata)
            r = "pearson"
        else:
            r_value, p_value = spearmanr(self.xdata, self.ydata)
            r = "spearman"
        self._name = self._names[r]
        self._test = r
        self._results.update({
            self._r_value: r_value,
            self._p_value: p_value,
            self._alpha_name: self._alpha
        })

    @property
    def r_value(self):
        """The correlation coefficient returned by the the determined test type"""
        return self._results[self._r_value]

    @property
    def statistic(self):
        return self._results[self._r_value]

    @property
    def test_type(self):
        """The test that was used to determine the correlation coefficient"""
        return self._test

    def __str__(self):
        out = list()

        out.append(std_output(self.name, self._results, [self._alpha_name, self._r_value, self._p_value]))
        out.append('')
        out.append(self._h0 if self.p_value > self._alpha else self._ha)
        out.append('')
        return '\n'.join(out)


class GroupComparison(Analysis):

    _min_size = 1
    _name = 'Group Comparison'
    _default_alpha = 0.05

    def __init__(self, xdata, ydata=None, groups=None, alpha=None, display=True):
        if ydata is None:
            if is_vector(xdata):
                vector = xdata
            else:
                raise AttributeError("ydata argument cannot be None.")
        else:
            vector = Vector(xdata, other=ydata, groups=groups)
        if vector.is_empty():
            raise NoDataError("Cannot perform test because there is no data")
        super(GroupComparison, self).__init__(vector, display=display)
        self._alpha = alpha or self._default_alpha
        self.logic()

    def run(self):
        raise NotImplementedError


class GroupCorrelation(GroupComparison):

    _names = {
        'pearson': 'Pearson Correlation Coefficient',
        'spearman': 'Spearman Correlation Coefficient',
    }
    _min_size = 2
    _r_value = 'r value'
    _p_value = 'p value'
    _group_name = 'Group'
    _n = 'n'

    def __init__(self, xdata, ydata=None, groups=None, alpha=None, display=True):
        self._test = None
        super(GroupCorrelation, self).__init__(xdata, ydata=ydata, groups=groups, alpha=alpha, display=display)

    def run(self):
        out = []
        # Remove any groups that are less than or equal to the minimum value from analysis.
        small_grps = [grp for grp, seq in self.data.groups.items() if len(seq) <= self._min_size]
        self.data.drop_groups(small_grps)
        if NormTest(*self.data.flatten(), display=False, alpha=self._alpha).p_value > self._alpha:
            r = "pearson"
            func = pearsonr
        else:
            r = 'spearman'
            func = spearmanr
        self._name = self._names[r]
        self._test = r
        for grp, pairs in self.data.paired_groups.items():
            r_value, p_value = func(*pairs)
            row_results = ({self._r_value: r_value,
                            self._p_value: p_value,
                            self._group_name: str(grp),
                            self._n: str(len(pairs[0]))})
            out.append(row_results)
        self._results = DataFrame(out).sort_values(self._group_name).to_dict(orient='records')

    def __str__(self):
        order = (
            self._n,
            self._r_value,
            self._p_value,
            self._group_name
        )
        return std_output(self._name, self._results, order=order)

    @property
    def counts(self):
        return tuple(s[self._n] for s in self._results)

    @property
    def r_value(self):
        return tuple(s[self._r_value] for s in self._results)

    @property
    def statistic(self):
        return tuple(s[self._r_value] for s in self._results)

    @property
    def p_value(self):
        return tuple(s[self._p_value] for s in self._results)


class GroupLinearRegression(GroupComparison):

    _name = "Linear Regression"
    _n = 'n'
    _slope = 'Slope'
    _intercept = 'Intercept'
    _r_value = 'r'
    _r_squared = 'r^2'
    _std_err = 'Std Err'
    _p_value = 'p value'
    _group_name = 'Group'

    def run(self):
        out = []
        # Remove any groups that are less than or equal to the minimum value from analysis.
        small_grps = [grp for grp, seq in self.data.groups.items() if len(seq) <= self._min_size]
        self.data.drop_groups(small_grps)
        for grp, pairs in self.data.paired_groups.items():
            slope, intercept, r, p_value, std_err = linregress(*pairs)
            count = len(pairs[0])
            out.append({
                self._n: str(count),
                self._slope: slope,
                self._intercept: intercept,
                self._r_value: r,
                self._r_squared: r ** 2,
                self._std_err: std_err,
                self._p_value: p_value,
                self._group_name: str(grp)
            })
        if not out:
            raise NoDataError
        self._results = DataFrame(out).sort_values(self._group_name).to_dict(orient='records')

    def __str__(self):
        order = (
            self._n,
            self._slope,
            self._intercept,
            self._r_squared,
            self._std_err,
            self._p_value,
            self._group_name
        )
        return std_output(self._name, self._results, order=order)

    @property
    def counts(self):
        return tuple(s[self._n] for s in self._results)

    @property
    def r_value(self):
        return tuple(s[self._r_value] for s in self._results)

    @property
    def statistic(self):
        return tuple(s[self._r_squared] for s in self._results)

    @property
    def p_value(self):
        return tuple(s[self._p_value] for s in self._results)

    @property
    def slope(self):
        return tuple(s[self._slope] for s in self._results)

    @property
    def intercept(self):
        return tuple(s[self._intercept] for s in self._results)

    @property
    def r_squared(self):
        return tuple(s[self._r_squared] for s in self._results)

    @property
    def std_err(self):
        return tuple(s[self._std_err] for s in self._results)
