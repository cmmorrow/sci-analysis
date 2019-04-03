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

    def __init__(self, sequence=None, name=None, order=None, dropna=False):
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
        """
        if sequence is None:
            self._values = pd.Series([])
            self._order = order
            self._name = name
            self._summary = pd.DataFrame([], columns=['counts', 'ranks', 'percents', 'categories'])
        elif is_data(sequence):
            new_name = sequence.name or name
            super(Categorical, self).__init__(v=sequence.data, n=new_name)
            self._order = sequence.order
            self._values = sequence.data
            self._name = sequence.name
            self._summary = sequence.summary
        else:
            self._name = name
            self._values = pd.Series(sequence)
            try:
                self._values.astype('category')
            except TypeError:
                self._values = pd.Series(flatten(sequence))
            except ValueError:
                self._values = pd.Series([])
            # Try to preserve the original dtype of the categories.
            try:
                if not any(self._values % 1):
                    self._values = self._values.astype(int)
            except TypeError:
                pass
            self._values = self._values.astype('category')

            if order is not None:
                if not is_iterable(order):
                    order = [order]
                self._values = self._values.cat.set_categories(order).cat.reorder_categories(order, ordered=True)
            if dropna:
                self._values = self._values.dropna()
            try:
                sequence += 1
                self._order = None if self._values.empty else self._values.cat.categories
            except TypeError:
                self._order = order
            counts = self._values.value_counts(sort=False, dropna=False, ascending=False)
            self._summary = pd.DataFrame({
                'counts': counts,
                'ranks': counts.rank(method='dense', na_option='bottom', ascending=False).astype('int'),
                'percents': (counts / counts.sum() * 100) if not all(counts == 0) else 0.0
            })
            self._summary['categories'] = self._summary.index.to_series()

            if order is not None:
                self._summary.sort_index(level=self._order, inplace=True, axis=0, na_position='last')
            else:
                self._summary.sort_values('ranks', inplace=True)
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
    def summary(self):
        return self._summary

    @property
    def counts(self):
        return self._summary.counts

    @property
    def percents(self):
        return self._summary.percents

    @property
    def order(self):
        return self._order

    @property
    def ranks(self):
        return self._summary.ranks

    @property
    def categories(self):
        return self._summary.categories

    @property
    def total(self):
        return len(self._values)

    @property
    def num_of_groups(self):
        return len(self._summary)
