# Scipy imports
from scipy.stats import (shapiro, kstest, ks_2samp, mannwhitneyu,
                         ttest_1samp, ttest_ind, f_oneway, kruskal, bartlett, levene)

from ..data import is_iterable, is_vector
from .base import Analysis, std_output
from .exc import NoDataError, MinimumSizeError


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
    _default_alpha = 0.05

    def __init__(self, *args, **kwargs):
        """Initialize the object"""

        from ..data import Vector
        self._alpha = kwargs['alpha'] if 'alpha' in kwargs else self._default_alpha
        if self._alpha is None:
            self._alpha = self._default_alpha
        display = kwargs['display'] if 'display' in kwargs else True
        data = Vector()
        for d in args:
            if not is_vector(d):
                if len(d) <= self._min_size:
                    raise MinimumSizeError("length of data is less than the minimum size {}.".format(self._min_size))
                data.append(Vector(d))
                if len(data) == 0:
                    raise NoDataError("Cannot perform test because there is no numeric data.")
            else:
                data = d
        if len(data) < 1:
            raise NoDataError("Cannot perform test because there is no numeric data.")
        if len(data) == 1:
            data = data[0]

        # set the _data and _display members
        super(Test, self).__init__(data, display=display)

        # Run the test and display the results
        self.logic()

    @property
    def statistic(self):
        """The test statistic returned by the function called in the run method."""
        return self._results['statistic']

    @property
    def statistic_name(self):
        """The test statistic name."""
        return self._statistic_name

    @property
    def p_value(self):
        """The p-value returned by the function called in the run method."""
        return self._results['p value']

    @property
    def alpha(self):
        """The alpha used by the hypothesis test."""
        return self._alpha

    def __str__(self):
        out = list()

        out.append(std_output(self.name, self._results, ['alpha', self.statistic_name, 'p value']))
        out.append('')
        out.append(self._h0 if self.p_value > self._alpha else self._ha)
        out.append('')
        return '\n'.join(out)

    def run(self):
        raise NotImplementedError


class NormTest(Test):
    """Tests for whether data is normally distributed or not."""

    _name = "Shapiro-Wilk test for normality"
    _statistic_name = 'W value'
    _h0 = "H0: Data is normally distributed"
    _ha = "HA: Data is not normally distributed"

    # TODO: Need to figure out how to perform the NormTest without generating a ton of new Vector objects.

    def run(self):
        w_value = list()
        p_value = list()
        for g, d in self._data.groups.items():
            _w, _p = shapiro(d)
            w_value.append(_w)
            p_value.append(_p)
        min_p = min(p_value)
        w_value = w_value[p_value.index(min_p)]
        p_value = min_p
        self._results.update({self._statistic_name: w_value, 'p value': p_value, 'alpha': self._alpha})

    @property
    def statistic(self):
        return self._results[self._statistic_name]

    @property
    def w_value(self):
        return self._results[self._statistic_name]


class KSTest(Test):
    """Tests whether data comes from a specified distribution or not."""

    _name = "Kolmogorov-Smirnov Test"
    _statistic_name = 'D value'

    def __init__(self, data, distribution='norm', parms=(), alpha=None, display=True):
        self._distribution = distribution
        self._parms = parms
        self._h0 = "H0: Data is matched to the " + self.distribution + " distribution"
        self._ha = "HA: Data is not from the " + self.distribution + " distribution"
        super(KSTest, self).__init__(data, alpha=alpha, display=display)

    def run(self):
        args = [self._data.data, self._distribution]
        if self._parms:
            args.append(self._parms)
        d_value, p_value = kstest(*args)
        self._results.update({self._statistic_name: d_value, 'p value': p_value, 'alpha': self._alpha})

    @property
    def distribution(self):
        """Return the distribution that data is being compared against"""
        return self._distribution

    @property
    def statistic(self):
        return self._results[self._statistic_name]

    @property
    def d_value(self):
        return self._results[self._statistic_name]


class TwoSampleKSTest(Test):
    """Tests whether two independent vectors come from the same distribution"""

    _name = "Two Sample Kolmogorov-Smirnov Test"
    _statistic_name = 'D value'
    _h0 = "H0: Both samples come from the same distribution"
    _ha = "HA: Samples do not come from the same distribution"

    def __init__(self, a, b=None, alpha=None, display=True):
        """

        Parameters
        ----------
        a : array-like or Vector
            One of the two vectors to compare.
        b : array-like or None
            The second vector to be compared against the first.
        alpha : float, optional
            The significance. Default is 0.05.
        display : bool, optional
            Print the results to stdout if True. Default is True.
        """
        if b is None:
            if is_vector(a):
                super(TwoSampleKSTest, self).__init__(a, alpha=alpha, display=display)
            else:
                raise AttributeError('second argument cannot be None.')
        else:
            super(TwoSampleKSTest, self).__init__(a, b, alpha=alpha, display=display)

    def run(self):
        args = self._data.groups.values()
        d_value, p_value = ks_2samp(*args)
        self._results.update({self._statistic_name: d_value, 'p value': p_value, 'alpha': self._alpha})

    @property
    def statistic(self):
        return self._results[self._statistic_name]

    @property
    def d_value(self):
        return self._results[self._statistic_name]


class MannWhitney(Test):
    """Performs a Mann Whitney U Test on two vectors"""

    _name = "Mann Whitney U Test"
    _statistic_name = 'u value'
    _h0 = "H0: Locations are matched"
    _ha = "HA: Locations are not matched"
    _min_size = 20

    def __init__(self, a, b=None, alpha=None, display=True):
        """

        Parameters
        ----------
        a : array-like or Vector
            One of the two vectors to compare.
        b : array-like or None
            The second vector to be compared against the first.
        alpha : float, optional
            The significance. Default is 0.05.
        display : bool, optional
            Print the results to stdout if True. Default is 0.05.
        """
        if b is None:
            if is_vector(a):
                super(MannWhitney, self).__init__(a, alpha=alpha, display=display)
            else:
                raise AttributeError('second argument cannot be None.')
        else:
            super(MannWhitney, self).__init__(a, b, alpha=alpha, display=display)

    def run(self):
        args = self._data.groups.values()
        if len(args) <= 1:
            raise NoDataError("At least one of the inputs is empty or non-numeric.")
        u_value, p_value = mannwhitneyu(*args, alternative='less')
        self._results.update({self._statistic_name: u_value, 'p value': p_value * 2, 'alpha': self._alpha})

    @property
    def statistic(self):
        return self._results[self._statistic_name]

    @property
    def u_value(self):
        return self._results[self._statistic_name]


class TTest(Test):
    """Performs a T-Test on the two provided vectors."""

    _names = {'1_sample': '1 Sample T Test', 't_test': 'T Test', 'welch_t': "Welch's T Test"}
    _statistic_name = 't value'
    _h0 = "H0: Means are matched"
    _ha = "HA: Means are significantly different"
    _min_size = 3

    def __init__(self, xdata, ydata=None, alpha=None, display=True):
        self._mu = None
        self._test = None
        if ydata is None:
            if is_vector(xdata):
                super(TTest, self).__init__(xdata, alpha=alpha, display=display)
            else:
                raise AttributeError('second argument cannot be None.')
        elif not is_iterable(ydata):
            self._mu = float(ydata)
            super(TTest, self).__init__(xdata, alpha=alpha, display=display)
        else:
            super(TTest, self).__init__(xdata, ydata, alpha=alpha, display=display)

    def run(self):
        if self._mu:
            t, p = ttest_1samp(self._data.data, self._mu, axis=0)
            test = "1_sample"
        else:
            num_args = len(self._data.groups.keys())
            if num_args <= 1:
                raise NoDataError("Cannot perform the test because there is no data")
            if EqualVariance(self._data, display=False, alpha=self._alpha).p_value > self._alpha:
                t, p = ttest_ind(*self._data.groups.values(), equal_var=True, axis=0)
                test = 't_test'
            else:
                t, p = ttest_ind(*self._data.groups.values(), equal_var=False, axis=0)
                test = 'welch_t'
        self._test = test
        self._name = self._names[test]
        self._results.update({'p value': p, self._statistic_name: float(t), 'alpha': self._alpha})

    @property
    def test_type(self):
        return self._test

    @property
    def mu(self):
        return self._mu

    @property
    def t_value(self):
        return self._results[self._statistic_name]

    @property
    def statistic(self):
        return self._results[self._statistic_name]


class Anova(Test):
    """Performs a one-way ANOVA on a group of vectors."""

    _name = "Oneway ANOVA"
    _statistic_name = 'f value'
    _h0 = "H0: Group means are matched"
    _ha = "HA: Group means are not matched"

    def run(self):
        if len(self._data.groups.values()) <= 1:
            raise NoDataError("Kruskal test requires at least tow numeric vectors.")
        f_value, p_value = f_oneway(*self.data.groups.values())
        self._results.update({'p value': p_value, self._statistic_name: f_value, 'alpha': self._alpha})

    @property
    def f_value(self):
        """The f value returned by the ANOVA f test"""
        return self._results[self._statistic_name]

    @property
    def statistic(self):
        return self._results[self._statistic_name]


class Kruskal(Test):
    """Performs a non-parametric Kruskal-Wallis test on a group of vectors."""

    _name = "Kruskal-Wallis"
    _statistic_name = 'h value'
    _h0 = "H0: Group means are matched"
    _ha = "HA: Group means are not matched"

    def run(self):
        if len(self._data.groups.values()) <= 1:
            raise NoDataError("Kruskal test requires at least tow numeric vectors.")
        h_value, p_value = kruskal(*self.data.groups.values())
        self._results.update({'p value': p_value, self._statistic_name: h_value, 'alpha': self._alpha})

    @property
    def h_value(self):
        """The h value returned by the Kruskal test"""
        return self._results[self._statistic_name]

    @property
    def statistic(self):
        return self._results[self._statistic_name]


class EqualVariance(Test):
    """Checks a group of vectors for equal variance."""

    _names = {'Bartlett': 'Bartlett Test', 'Levene': 'Levene Test'}
    _statistic_name = {'Bartlett': 'T value', 'Levene': 'W value'}
    _h0 = "H0: Variances are equal"
    _ha = "HA: Variances are not equal"

    def __init__(self, *data, **kwargs):
        self._test = None
        super(EqualVariance, self).__init__(*data, **kwargs)

    def run(self):
        if len(self._data) < self._min_size:
            pass
        if len(self._data.groups.values()) <= 1:
            raise NoDataError("Equal variance test requires at least two numeric vectors.")
        if NormTest(self._data, display=False, alpha=self._alpha).p_value > self._alpha:
            statistic, p_value = bartlett(*self._data.groups.values())
            r = 'Bartlett'
            self._results.update({'p value': p_value, self._statistic_name[r]: statistic, 'alpha': self._alpha})
        else:
            statistic, p_value = levene(*self._data.groups.values())
            r = 'Levene'
            self._results.update({'p value': p_value, self._statistic_name[r]: statistic, 'alpha': self._alpha})
        self._test = r
        self._name = self._names[r]

    @property
    def t_value(self):
        return self._results[self._statistic_name['Bartlett']]

    @property
    def w_value(self):
        return self._results[self._statistic_name['Levene']]

    @property
    def statistic(self):
        try:
            s = self._results[self._statistic_name['Levene']]
        except KeyError:
            s = self._results[self._statistic_name['Bartlett']]
        return s

    @property
    def statistic_name(self):
        return self._statistic_name[self._test]

    @property
    def test_type(self):
        """The test that was used to check for equal variance"""
        return self._test
