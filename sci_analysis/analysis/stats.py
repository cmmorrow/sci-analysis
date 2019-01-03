from math import sqrt

# Pandas imports
from pandas import DataFrame

# Numpy imports
from numpy import mean, std, median, amin, amax, percentile

# Scipy imports
from scipy.stats import skew, kurtosis, sem

from .base import Analysis, std_output
from .exc import NoDataError, MinimumSizeError
from ..data import Vector, Categorical, is_dict, is_group, is_categorical, is_vector, is_tuple


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
        d = Vector(data)
        if d.is_empty():
            raise NoDataError("Cannot perform the test because there is no data")
        if len(d) <= self._min_size:
            raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))

        super(VectorStatistics, self).__init__(d, display=display)
        self.logic()

    def run(self):
        dof = 1 if self._sample else 0
        vmin = amin(self._data.data)
        vmax = amax(self._data.data)
        vrange = vmax - vmin
        q1 = percentile(self._data.data, 25)
        q3 = percentile(self._data.data, 75)
        iqr = q3 - q1
        self._results = {self._n: len(self._data.data),
                         self._mean: mean(self._data.data),
                         self._std: std(self._data.data, ddof=dof),
                         self._ste: sem(self._data.data, 0, dof),
                         self._q2: median(self._data.data),
                         self._min: vmin,
                         self._max: vmax,
                         self._range: vrange,
                         self._skew: skew(self._data.data),
                         self._kurt: kurtosis(self._data.data),
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
    _total = 'Total'
    _pooled = 'Pooled Std Dev'
    _gmean = 'Grand Mean'
    _gmedian = 'Grand Median'
    _num_of_groups = 'Number of Groups'

    def __init__(self, *args, **kwargs):
        groups = kwargs.get('groups', None)
        display = kwargs.get('display', False)
        if is_dict(args[0]):
            _data, = args
        elif is_group(args,):
            _data = dict(zip(groups, args)) if groups else dict(zip(list(range(1, len(args) + 1)), args))
        else:
            _data = None
        data = Vector()
        for g, d in _data.items():
            if len(d) == 0:
                raise NoDataError("Cannot perform test because there is no data")
            if len(d) <= self._min_size:
                raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
            data.append(Vector(d, groups=[g for _ in range(0, len(d))]))
        if data.is_empty():
            raise NoDataError("Cannot perform test because there is no data")
        self.k = None
        self.total = None
        self.pooled = None
        self.gmean = None
        self.gmedian = None
        super(GroupStatistics, self).__init__(data, display=display)
        self.logic()

    def logic(self):
        if not self._data:
            pass
        self._results = []
        self.run()
        if self._display:
            print(self)

    def run(self):
        out = []
        for group, vector in self._data.groups.items():
            row_result = {self._group: str(group),
                          self._n: len(vector),
                          self._mean: mean(vector),
                          self._std: std(vector, ddof=1),
                          self._max: amax(vector),
                          self._q2: median(vector),
                          self._min: amin(vector),
                          }
            out.append(row_result)
        summ = DataFrame(out).sort_values(self._group)
        self.total = len(self._data.data)
        self.k = len(summ)
        if self.k > 1:
            self.pooled = sqrt(((summ[self._n] - 1) * summ[self._std] ** 2).sum() / (summ[self._n].sum() - self.k))
            self.gmean = summ[self._mean].mean()
            self.gmedian = median(summ[self._q2])
            self._results = ({
                self._num_of_groups: self.k,
                self._total: self.total,
                self._pooled: self.pooled,
                self._gmean: self.gmean,
                self._gmedian: self.gmedian,
            }, summ)
        else:
            self._results = summ

    def __str__(self):
        order = (
            self._num_of_groups,
            self._total,
            self._gmean,
            self._pooled,
            self._gmedian,
        )
        group_order = (
            self._n,
            self._mean,
            self._std,
            self._min,
            self._q2,
            self._max,
            self._group,
        )
        if is_tuple(self._results):
            out = '{}\n{}'.format(
                std_output('Overall Statistics', self._results[0], order=order),
                std_output(self._name, self._results[1].to_dict(orient='records'), order=group_order),
            )
        else:
            out = std_output(self._name, self._results.to_dict(orient='records'), order=group_order)
        return out

    @property
    def grand_mean(self):
        return self.gmean

    @property
    def grand_median(self):
        return self.gmedian

    @property
    def pooled_std(self):
        return self.pooled


class GroupStatisticsStacked(Analysis):

    _min_size = 1
    _name = 'Group Statistics'
    _agg_name = 'Overall Statistics'
    _group = 'Group'
    _n = 'n'
    _mean = 'Mean'
    _std = 'Std Dev'
    _max = 'Max'
    _q2 = 'Median'
    _min = 'Min'
    _total = 'Total'
    _pooled = 'Pooled Std Dev'
    _gmean = 'Grand Mean'
    _gmedian = 'Grand Median'
    _num_of_groups = 'Number of Groups'

    def __init__(self, values, groups=None, **kwargs):
        display = kwargs['display'] if 'display' in kwargs else True
        if groups is None:
            if is_vector(values):
                data = values
            else:
                raise AttributeError('ydata argument cannot be None.')
        else:
            data = Vector(values, groups=groups)
        if data.is_empty():
            raise NoDataError("Cannot perform test because there is no data")
        self.pooled = None
        self.gmean = None
        self.gmedian = None
        self.total = None
        self.k = None
        super(GroupStatisticsStacked, self).__init__(data, display=display)
        self.logic()

    def logic(self):
        if not self._data:
            pass
        self._results = []
        self.run()
        if self._display:
            print(self)

    def run(self):
        out = []
        for group, vector in self._data.groups.items():
            if len(vector) <= self._min_size:
                raise MinimumSizeError("length of data is less than the minimum size {}".format(self._min_size))
            row_result = {self._group: group,
                          self._n: len(vector),
                          self._mean: mean(vector),
                          self._std: std(vector, ddof=1),
                          self._max: amax(vector),
                          self._q2: median(vector),
                          self._min: amin(vector),
                          }
            out.append(row_result)
        summ = DataFrame(out).sort_values(self._group)
        self.total = len(self._data.data)
        self.k = len(summ)
        if self.k > 1:
            self.pooled = sqrt(((summ[self._n] - 1) * summ[self._std] ** 2).sum() / (summ[self._n].sum() - self.k))
            self.gmean = summ[self._mean].mean()
            self.gmedian = median(summ[self._q2])
            self._results = ({
                self._num_of_groups: self.k,
                self._total: self.total,
                self._pooled: self.pooled,
                self._gmean: self.gmean,
                self._gmedian: self.gmedian,
            }, summ)
        else:
            self._results = summ

    def __str__(self):
        order = (
            self._num_of_groups,
            self._total,
            self._gmean,
            self._pooled,
            self._gmedian,
        )
        group_order = (
            self._n,
            self._mean,
            self._std,
            self._min,
            self._q2,
            self._max,
            self._group,
        )
        if is_tuple(self._results):
            out = '{}\n{}'.format(
                std_output(self._agg_name, self._results[0], order=order),
                std_output(self._name, self._results[1].to_dict(orient='records'), order=group_order),
            )
        else:
            out = std_output(self._name, self._results.to_dict(orient='records'), order=group_order)
        return out

    @property
    def grand_mean(self):
        return self.gmean

    @property
    def grand_median(self):
        return self.gmedian

    @property
    def pooled_std(self):
        return self.pooled


class CategoricalStatistics(Analysis):
    """Reports basic summary stats for Categorical data."""

    _min_size = 1
    _name = 'Statistics'
    _agg_name = 'Overall Statistics'
    _rank = 'Rank'
    _cat = 'Category'
    _freq = 'Frequency'
    _perc = 'Percent'
    _total = 'Total'
    _num_of_grps = 'Number of Groups'

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
        col = dict(categories=self._cat,
                   counts=self._freq,
                   percents=self._perc,
                   ranks=self._rank)
        self.data.summary.rename(columns=col, inplace=True)
        if self.data.num_of_groups > 1:
            self._results = ({
                self._total: self.data.total,
                self._num_of_grps: self.data.num_of_groups,
            }, self.data.summary.to_dict(orient='records'))
        else:
            self._results = self.data.summary.to_dict(orient='records')

    def __str__(self):
        order = (
            self._total,
            self._num_of_grps,
        )
        grp_order = (
            self._rank,
            self._freq,
            self._perc,
            self._cat,
        )
        if is_tuple(self._results):
            out = '{}\n{}'.format(
                std_output(self._agg_name, self._results[0], order=order),
                std_output(self._name, self._results[1], order=grp_order),
            )
        else:
            out = std_output(self._name, self._results, order=grp_order)
        return out
