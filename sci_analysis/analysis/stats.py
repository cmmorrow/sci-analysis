# Pandas imports
from pandas import DataFrame

# Numpy imports
from numpy import mean, std, median, amin, amax, percentile

# Scipy imports
from scipy.stats import skew, kurtosis, sem

from .base import Analysis, std_output
from .exc import NoDataError, MinimumSizeError
from ..data import Vector, Categorical, is_dict


class VectorStatistics(Analysis):
    """Reports basic summary stats for a provided vector."""

    _min_size = 1
    _name = 'Statistics'

    def __init__(self, data, sample=True, display=True):
        self._sample = sample
        d = Vector(data).data_prep()
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
        order = ['Count',
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
                 'Range']
        return std_output(self._name, results=self._results, order=order)


class GroupStatistics(Analysis):
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
            clean = Vector(d).data_prep()
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
        order = [
            'Count',
            'Mean',
            'Std Dev',
            'Min',
            'Median',
            'Max',
            'Group'
        ]
        return std_output(self._name, self._results, order=order)


class CategoricalStatistics(Analysis):
    """Reports basic summary stats for Categorical data."""

    _min_size = 1
    _name = 'Statistics'

    def __init__(self, data, display=True):
        self.total = None
        d = Categorical(data)
        if d.data.empty or len(d.categories) == 0:
            raise NoDataError("Cannot perform the test because there is no data")

        super(CategoricalStatistics, self).__init__(d, display=display)
        self.logic()

    def run(self):
        results = list()
        ranks = self.data.counts.rank(method='first', na_option='bottom', ascending=False).astype('int')
        df = DataFrame({'Rank': ranks,
                        'Category': self.data.counts.index,
                        'Frequency': self.data.counts})
        for _, row in df.sort_values('Rank').iterrows() if self.data.order is None else df.iterrows():
            results.append(row.to_dict())
        self._results = results

    def __str__(self):
        order = [
            'Rank',
            'Frequency',
            'Category'
        ]
        return std_output(self._name, self._results, order=order)
