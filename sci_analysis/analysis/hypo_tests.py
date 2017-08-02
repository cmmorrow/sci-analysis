# Scipy imports
from scipy.stats import shapiro, kstest, ks_2samp, mannwhitneyu, ttest_1samp, ttest_ind, f_oneway, kruskal, bartlett, \
    levene

from data import is_group, is_iterable
from analysis.base import Analysis
try:
    from analysis.base import std_output
except ImportError:
    pass
from analysis.func import NoDataError, MinimumSizeError


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

        from data import Vector
        self._alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05
        data = list()
        for d in args:
            clean = Vector(d).data_prep()
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

    _names = {'1_sample': '1 Sample T Test', 't_test': 'T Test', 'welch_t': "Welch's T Test"}
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
        self._name = self._names[test]
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

    _names = {'Bartlett': 'Bartlett Test', 'Levene': 'Levene Test'}
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
            r = 'Bartlett'
            self._results.update({'p value': p_value, 'T value': statistic})
        else:
            statistic, p_value = levene(*self._data)
            r = 'Levene'
            self._results.update({'p value': p_value, 'W value': statistic})
        self._test = r
        self._name = self._names[r]

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
