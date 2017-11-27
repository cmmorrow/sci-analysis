# Scipy imports
from scipy.stats import linregress, pearsonr, spearmanr

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

    def __init__(self, xdata, ydata=None, alpha=0.05, display=True):
        self._alpha = alpha
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
    _h0 = "H0: There is no significant relationship between predictor and response"
    _ha = "HA: There is a significant relationship between predictor and response"

    def __init__(self, xdata, ydata=None, alpha=0.05, display=True):
        super(LinearRegression, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        slope, intercept, r, p_value, std_err = linregress(self.xdata, self.ydata)
        count = len(self.xdata)
        self._results.update({'Count': count,
                              'Slope': slope,
                              'Intercept': intercept,
                              'r': r,
                              'r^2': r ** 2,
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
        return self._results['r^2']

    @property
    def r_value(self):
        return self._results['r']

    @property
    def statistic(self):
        return self._results['r^2']

    @property
    def std_err(self):
        return self._results['Std Err']

    def __str__(self):
        """If the result is greater than the significance, print the null hypothesis, otherwise,
        the alternate hypothesis"""
        out = list()
        order = [
            'Count',
            'Slope',
            'Intercept',
            'r',
            'r^2',
            'Std Err',
            'p value'
        ]
        out.append(std_output(self._name, self._results, order=order))
        out.append('')
        out.append(self._h0 if self.p_value > self._alpha else self._ha)
        out.append('')
        return '\n'.join(out)


class Correlation(Comparison):
    """Performs a pearson or spearman correlation between two vectors."""

    _names = {'pearson': 'Pearson Correlation Coefficient', 'spearman': 'Spearman Correlation Coefficient'}
    _h0 = "H0: There is no significant relationship between predictor and response"
    _ha = "HA: There is a significant relationship between predictor and response"

    def __init__(self, xdata, ydata=None, alpha=0.05, display=True):
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
        self._results.update({'r value': r_value, 'p value': p_value, 'alpha': self._alpha})

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
        out = list()

        out.append(std_output(self.name, self._results, ['alpha', 'r value', 'p value']))
        out.append('')
        out.append(self._h0 if self.p_value > self._alpha else self._ha)
        out.append('')
        return '\n'.join(out)
