from warnings import warn

# Import packages
import pandas as pd

# Import from local
from data import Data, flatten, is_data


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
            self._counts = pd.Series([])
            self._order = order
            self._name = name
        elif is_data(sequence):
            new_name = sequence.name or name
            super(Categorical, self).__init__(v=sequence.data, n=new_name)
            self._order = sequence.order
        else:
            cat_kwargs = {'dtype': 'category'}
            if order is not None:
                cat_kwargs.update({'categories': order, 'ordered': True})
            try:
                self._values = pd.Series(sequence).astype(**cat_kwargs)
            except TypeError:
                self._values = pd.Series(flatten(sequence)).astype(**cat_kwargs)
            except ValueError:
                self._values = pd.Series([])
            self._name = name
            if dropna:
                self._values = self._values.dropna()
            try:
                # TODO: Need to fix this to work with numeric lists
                sequence += 1
                self._order = self.categories
            except TypeError:
                self._order = order
        self._counts = self._values.value_counts(sort=False, dropna=False)
        if self.categories is not None:
            if len(self.categories) > NumberOfCategoriesWarning.warn_categories:
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

    def data_prep(self):
        return self._values.dropna().reset_index(drop=True)

    @property
    def data_type(self):
        return self.data.dtype

    @property
    def counts(self):
        return self._counts

    @property
    def order(self):
        return self._order

    @property
    def categories(self):
        # TODO: Need to fix this to show NaN since Pandas will drop NaN automatically.
        return self._values.cat.categories if len(self._values) > 0 else None
