"""sci_analysis module: data
Functions:
    assign: convert the passed array_like arguments to the correct sci_analysis object.
    is_vector: test if the passed array_like argument is a sci_analysis Vector object.
    is_data: test if the passed array_like argument is a sci_analysis Data object.
"""
from __future__ import absolute_import
from warnings import warn
# Import packages
import pandas as pd

# Import from local
from sci_analysis.operations.data_operations import is_dict, is_iterable, flatten


class NumberOfCategoriesWarning(Warning):

    warn_categories = 50

    def __str__(self):
        return "The number of categories is greater than {} which might make analysis difficult. " \
               "If this isn't a mistake, consider subsetting the data first".format(self.warn_categories)


# def assign(obj, other=None):
#     """
#     Convert the passed array_like arguements to the correct sci_analysis object.
#
#     Parameters
#     ----------
#     obj : object
#         The input object.
#     other : object of unknown type, optional
#         A secondary object corresponding to the input object
#
#     Returns
#     -------
#     subseq : Data
#         A sci_analysis object that inherits from Data
#     """
#     return (Vector(obj), Vector(other)) if other is not None else Vector(obj)


def is_vector(obj):
    """
    Test if the passed array_like argument is a sci_analysis Vector object.

    Parameters
    ----------
    obj : object
        The input object.

    Returns
    -------
    test result : bool
        The test result of whether seq is a sci_analysis Vector object or not.
    """
    return isinstance(obj, Vector)


def is_data(obj):
    """
    Test if the passed array_like argument is a sci_analysis Data object.

    Parameters
    ----------
    obj : object
        The input object.

    Returns
    -------
    test result : bool
        The test result of whether seq is a sci_analysis Data object or not.
    """
    return isinstance(obj, Data)


def is_numeric(obj):
    """
    Test if the passed array_like argument is a sci_analysis Numeric object.

    Parameters
    ----------
    obj : object
        The input object.

    Returns
    -------
    test result : bool
        The test result of whether seq is a sci_analysis Numeric object or not.
    """
    return isinstance(obj, Numeric)


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


class EmptyVectorError(Exception):
    """
    Exception raised when the length of a Vector object is 0.
    """
    pass


class UnequalVectorLengthError(Exception):
    """
    Exception raised when the length of two Vector objects are not equal, i.e., len(Vector1) != len(Vector2)
    """
    pass


class Data(object):
    """
    The super class used by all objects representing data for analysis
    in sci_analysis. All analysis classes should expect the data provided through
    arguments to be a descendant of this class.

    Data members are data_type, data and name. data_type is used for identifying
    the container class. The data member stores the data provided through an
    argument. The name member is an optional name for the Data object.
    """

    def __init__(self, v=None, n=None):
        """
        Sets the data and name members.

        Parameters
        ----------
        v : array_like
            The input object
        n : str
            The name of the Data object
        """
        self._values = v
        self._name = n

    def is_empty(self):
        """
        Tests if this Data object's data member equals 'None' and returns the result.

        Returns
        -------
        test result : bool
            The result of whether self._values is set or not
        """
        return self._values is None

    @property
    def data(self):
        return self._values

    @property
    def name(self):
        return self._name

    def __repr__(self):
        """
        Prints the Data object using the same representation as its data member.

        Returns
        -------
        output : str
            The string representation of the encapsulated data.
        """
        return self._values.__repr__()

    def __len__(self):
        """Returns the length of the data member. If data is not defined, 0 is returned. If the data member is a scalar
        value, 1 is returned.

        Returns
        -------
        length : int
            The length of the encapsulated data.
        """
        if self._values is not None:
            try:
                return len(self._values)
            except TypeError:
                return 1
        else:
            return 0

    def __getitem__(self, item):
        """
        Gets the value of the data member at index item and returns it.

        Parameters
        ----------
        item : int
            An index of the encapsulating data.

        Returns
        -------
        value : object
            The value of the encapsulated data at the specified index, otherwise None if no such index exists.
        """
        try:
            return self._values[item]
        except (IndexError, AttributeError):
            return None

    def __contains__(self, item):
        """
        Tests whether the encapsulated data contains the specified index or not.

        Parameters
        ----------
        item : int
            An index of the encapsulating data.

        Returns
        -------
        test result : bool
            The test result of whether item is a valid index of the encapsulating data or not.
        """
        try:
            return item in self._values
        except AttributeError:
            return None

    def __iter__(self):
        """
        Give this Data object the iterative behavior of its encapsulated data.

        Returns
        -------
        itr :iterator
            An iterator based on the encapsulated sequence.
        """
        return self._values.__iter__()


class Numeric(Data):
    """An abstract class that all Data classes that represent numeric data should inherit from."""

    def __init__(self, sequence=None, name=None):
        """Takes an array-like object and converts it to a pandas Series with any non-numeric values converted to NaN.

        Parameters
        ----------
        sequence : array-like
            The input object
        name : str, optional
            The name of the Numeric object
        """
        if sequence is None:
            self._values = pd.Series([])
            self._type = None
        elif is_data(sequence):
            super(Numeric, self).__init__(v=sequence.data, n=name)
            self._type = self._values.dtype
        elif is_iterable(sequence):
            if hasattr(sequence, 'shape'):
                if len(sequence.shape) > 1:
                    sequence = sequence.flatten()
            elif not is_dict(sequence):
                sequence = flatten(sequence)
            self._values = pd.to_numeric(pd.Series(sequence), errors='coerce')
            self._type = self._values.dtype
        else:
            try:
                self._values = pd.to_numeric(pd.Series([sequence], index=[0]), errors='coerce')
            except (ValueError, TypeError):
                self._values = pd.Series([])
            self._type = None
        self._name = name

    def data_prep(self):
        """
        Converts the values of _name to conform to the Data object standards.

        Returns
        -------
        data : np.array
            The enclosed data represented as a numpy array.
        """
        raise NotImplementedError

    def drop_nan(self):
        """
        Removes NaN values from the Numeric object and returns the resulting pandas Series. The length of the returned
        object is the original object length minus the number of NaN values removed from the object.

        Returns
        -------
        arr : pandas.Series
            A copy of the Numeric object's internal Series with all NaN values removed.
        """
        return self._values.dropna().reset_index(drop=True)

    def drop_nan_intersect(self, seq):
        """
        Removes the value from the internal Vector object and seq at i where i is nan in the internal Vector object or
        seq.

        Parameters
        ----------
        seq : array-like
            A corresponding sequence.

        Returns
        -------
        arr1, arr2 : tuple
            A tuple of numpy Arrays corresponding to the internal Vector and seq with all nan values removed.
        """
        c = pd.DataFrame({'self': self._values, 'other': seq}).dropna().reset_index(drop=True)
        return c['self'], c['other']


class Vector(Numeric):
    """
    The sci_analysis representation of continuous, numeric data.
    """

    def __init__(self, sequence=None, name=None):
        """
        Takes an array-like object and converts it to a pandas Series of
        dtype float64, with any non-numeric values converted to NaN.

        Parameters
        ----------
        sequence : array-like
            The input object
        name : str, optional
            The name of the Vector object
        """

        super(Vector, self).__init__(sequence=sequence, name=name)
        if not self._values.empty:
            self._values = self._values.astype('float')
            if self._values.dtype != self._type:
                self._type = self._values.dtype

    @property
    def data_type(self):
        return self._type

    def data_prep(self, other=None):
        """
        Remove all nan values from the encapsulated Array.

        Parameters
        ----------
        other : array-like, optional
            A secondary array-like object with corresponding NaN values to remove.

        Returns
        -------
        vector : Vector
            A vector object with all nan values removed.
        """
        if other is not None:
            vector = other if is_vector(other) else Vector(other)
            if len(self.data) != len(vector.data):
                raise UnequalVectorLengthError("x and y data lengths are not equal")

            x, y = self.drop_nan_intersect(vector)

            if len(x) == 0 or len(y) == 0:
                return None
                # raise EmptyVectorError("Passed data is empty")
            return x, y
        elif not is_iterable(self.data):
            return pd.Series(self.data).astype(float)
        else:
            v = self.drop_nan()
            if len(v) == 0:
                return None
                # raise EmptyVectorError("Passed data is empty")
            return v

    def is_empty(self):
        """
        Overrides the super class's method to also check for length of zero.

        Returns
        -------
        test_result : bool
            The result of whether the length of the Vector object is 0 or not.
        """
        # return len(self._values) == 0
        return self._values.empty


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
        # self._total = len(self.categories) if self.categories is not None else 0.0

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
