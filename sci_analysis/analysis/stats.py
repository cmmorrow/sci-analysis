# Pandas imports
from pandas import DataFrame

# Numpy imports
from numpy import mean, std, median, amin, amax, percentile

# Scipy imports
from scipy.stats import skew, kurtosis, sem

from .base import Analysis, std_output
from .exc import NoDataError, MinimumSizeError
from ..data import Vector, Categorical, is_dict, is_categorical


class VectorStatistics(Analysis):
    """Reports basic summary stats for a provided vector."""

    _min_size = 1
    _name = 'Statistics'
    _n = 'n'
    _mean = 'Mean'
    _std = 'Std Dev'
    _ste = 'Std Error'
    _range = 'Range'
    _skew = 'Skewness'
    _kurt = 'Kurtosis'
    _iqr = 'IQR'
    _q1 = '25%'
    _q2 = '50%'
    _q3 = '75%'
    _min = 'Minimum'
    _max = "Maximum"

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
        self._results = {self._n: count,
                         self._mean: avg,
                         self._std: sd,
                         self._ste: error,
                         self._q2: med,
                         self._min: vmin,
                         self._max: vmax,
                         self._range: vrange,
                         self._skew: sk,
                         self._kurt: kurt,
                         self._q1: q1,
                         self._q3: q3,
                         self._iqr: iqr,
                         }

    @property
    def count(self):
        return self._results[self._n]

    @property
    def mean(self):
        return self._results[self._mean]

    @property
    def std_dev(self):
        return self._results[self._std]

    @property
    def std_err(self):
        return self._results[self._ste]

    @property
    def median(self):
        return self._results[self._q2]

    @property
    def minimum(self):
        return self._results[self._min]

    @property
    def maximum(self):
        return self._results[self._max]

    @property
    def range(self):
        return self._results[self._range]

    @property
    def skewness(self):
        return self._results[self._skew]

    @property
    def kurtosis(self):
        return self._results[self._kurt]

    @property
    def q1(self):
        return self._results[self._q1]

    @property
    def q3(self):
        return self._results[self._q3]

    @property
    def iqr(self):
        return self._results[self._iqr]

    def __str__(self):
        order = [self._n,
                 self._mean,
                 self._std,
                 self._ste,
                 self._skew,
                 self._kurt,
                 self._max,
                 self._q3,
                 self._q2,
                 self._q1,
                 self._min,
                 self._iqr,
                 self._range,
                 ]
        return std_output(self._name, results=self._results, order=order)


class GroupStatistics(Analysis):
    """Reports basic summary stats for a group of vectors."""

    _min_size = 1
    _name = 'Group Statistics'
    _group = 'Group'
    _n = 'n'
    _mean = 'Mean'
    _std = 'Std Dev'
    _max = 'Max'
    _q2 = 'Median'
    _min = 'Min'

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
            row_result = {self._group: group,
                          self._n: count,
                          self._mean: avg,
                          self._std: sd,
                          self._max: vmax,
                          self._q2: q2,
                          self._min: vmin,
                          }
            self._results.append(row_result)

    def __str__(self):
        order = [
            self._n,
            self._mean,
            self._std,
            self._min,
            self._q2,
            self._max,
            self._group,
        ]
        return std_output(self._name, self._results, order=order)


class CategoricalStatistics(Analysis):
    """Reports basic summary stats for Categorical data."""

    _min_size = 1
    _name = 'Statistics'
    _rank = 'Rank'
    _cat = 'Category'
    _freq = 'Frequency'
    _perc = 'Percent'

    def __init__(self, data, **kwargs):
        order = kwargs['order'] if 'order' in kwargs else None
        dropna = kwargs['dropna'] if 'dropna' in kwargs else False
        display = kwargs['display'] if 'display' in kwargs else True
        self.ordered = True if order is not None else False
        d = data if is_categorical(data) else Categorical(data, order=order, dropna=dropna)
        if d.is_empty():
            raise NoDataError("Cannot perform the test because there is no data")

        super(CategoricalStatistics, self).__init__(d, display=display)
        self.logic()

    def run(self):
        results = list()
        self.data.summary.rename(columns={'categories': self._cat,
                                          'counts': self._freq,
                                          'percents': self._perc,
                                          'ranks': self._rank}, inplace=True)
        for _, row in self.data.summary.iterrows():
            results.append(row.to_dict())
        self._results = results

    def __str__(self):
        order = [
            self._rank,
            self._freq,
            self._perc,
            self._cat,
        ]
        return std_output(self._name, self._results, order=order)
