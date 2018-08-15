from warnings import warn

# Import packages
import pandas as pd

# Import from local
from .data import Data, is_data
from .data_operations import flatten, is_iterable


class NumberOfCategoriesWarning(Warning):

    warn_categories = 50

    def __str__(self):
        return "The number of categories is greater than {} which might make analysis difficult. " \
               "If this isn't a mistake, consider subsetting the data first".format(self.warn_categories)


def is_categorical(obj):
    """
    Test if the passed array_like argument is a sci_analysis Categorical object.

    Parameters
    ----------
    obj : object
        The input object.

    Returns
    -------
    test result : bool
        The test result of whether obj is a sci_analysis Categorical object.
    """
    return isinstance(obj, Categorical)


class Categorical(Data):
    """
    The sci_analysis representation of categorical, quantitative or textual data.
    """

    _ind = 'ind'
    _grp = 'groups'
    _cnt = 'counts'
    _rnk = 'ranks'
    _pct = 'percents'
    _cat = 'categories'
    _col_names = (_ind, _grp)

    def __init__(self, sequence=None, name=None, order=None, dropna=False, groups=None):
        """Takes an array-like object and converts it to a pandas Categorical object.

        Parameters
        ----------
        sequence : array-like or Data or Categorical
            The input object.
        name : str, optional
            The name of the Categorical object.
        order : array-like
            The order that categories in sequence should appear.
        dropna : bool
            Remove all occurances of numpy NaN.
        groups : list | set | tuple | np.array | pd.Series, optional
            The sequence of group names for sub-arrays
        """
        self._summ_col_names = (self._cnt, self._rnk, self._pct, self._cat)
        self._values = pd.DataFrame([], columns=self._col_names)
        self._summary = pd.DataFrame([], columns=self._summ_col_names)

        if sequence is None:
            self._order = order
            self._name = name
            self._groups = groups
        elif is_data(sequence):
            new_name = sequence.name or name
            super(Categorical, self).__init__(v=sequence.values, n=new_name)
            self._order = sequence.order
            self._summary = sequence.summary
        else:
            self._name = name
            try:
                self._values[self._ind] = pd.Series(sequence).reset_index(drop=True).astype('category')
            except TypeError:
                self._values[self._ind] = pd.Series(flatten(sequence)).astype('category')
            except ValueError:
                self._values = pd.Series([])
            try:
                self._values[self._grp] = 1 if groups is None else groups
            except ValueError:
                raise AttributeError('The length of groups must be equal to the length of sequence.')
            self._values.loc[:, self._grp] = self._values[self._grp].astype('category')
            if order is not None:
                if not is_iterable(order):
                    order = [order]
                self._values[self._ind] = (
                    self._values[self._ind]
                        .cat.set_categories(order)
                        .cat.reorder_categories(order, ordered=True)
                )
            if dropna:
                self._values = self._values.dropna()
            try:
                sequence += 1
                self._order = None if self._values.empty else self._values[self._ind].cat.categories
            except TypeError:
                self._order = order
            if len(self._values[self._grp].unique()) > 1:
                self._values['agg'] = 1
                counts = (
                    self._values
                        .groupby([self._ind, self._grp])
                        .count()
                        .fillna(0)
                        .rename(columns={'agg': self._ind})[self._ind]
                )
            else:
                counts = self._values[self._ind].value_counts(sort=False, dropna=False, ascending=False)
            ranks = counts.rank(method='dense', na_option='bottom', ascending=False).astype('int')
            percents = (counts / counts.sum() * 100) if not all((counts == 0).tolist()) else 0.0
            self._summary = pd.DataFrame({
                self._cnt: counts,
                self._rnk: ranks,
                self._pct: percents
            })
            if len(self._values[self._grp].unique()) > 1:
                self._summary[self._cat] = self._summary.index.to_series().apply(lambda c: ', '.join(map(str, c)))
            else:
                self._summary[self._cat] = self._summary.index.to_series()
            if order is not None:
                self._summary.sort_index(level=0, inplace=True, axis=0, na_position='last')
            else:
                self._summary.sort_values(self._rnk, inplace=True)
        if not self._summary.empty and len(self.categories) > NumberOfCategoriesWarning.warn_categories:
            warn(NumberOfCategoriesWarning())

    def is_empty(self):
        """
        Overrides the super class's method to also check for length of zero.

        Returns
        -------
        test_result : bool
            The result of whether the length of the Vector object is 0 or not.
        """
        return self._values.empty

    @property
    def data(self):
        return self._values[self._ind]

    @property
    def values(self):
        return self._values

    @property
    def summary(self):
        return self._summary

    @property
    def counts(self):
        return self._summary[self._cnt]

    @property
    def percents(self):
        return self._summary[self._pct]

    @property
    def order(self):
        return self._order

    @property
    def ranks(self):
        return self._summary[self._rnk]

    @property
    def categories(self):
        return self._summary[self._cat]

    @property
    def total(self):
        return len(self._values)

    @property
    def groups(self):
        return self._values[self._grp]

    @property
    def group_names(self):
        return self._values[self._grp].unique().dropna()

    @property
    def num_of_categories(self):
        return len(self._summary)

    @property
    def has_groups(self):
        return len(self.group_names) > 1
