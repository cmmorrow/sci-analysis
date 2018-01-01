# Import packages
import pandas as pd
import numpy as np
import datetime

# Import from local
from .data import Data, is_data
from .data_operations import is_iterable, flatten


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


class Numeric(Data):
    """An abstract class that all Data classes that represent numeric data should inherit from."""

    _col_names = ['ind', 'dep', 'grp']

    def __init__(self, sequence=None, other=None, groups=None, name=None):
        """Takes an array-like object and converts it to a pandas Series with any non-numeric values converted to NaN.

        Parameters
        ----------
        sequence : array-like or int or float
            The input object
        other : array-like
            The secondary input object
        groups : array-like
            The sequence of group names for sub-arrays
        name : str, optional
            The name of the Numeric object
        """
        self._auto_groups = True if groups is None else False
        if sequence is None:
            super(Numeric, self).__init__(v=pd.DataFrame([], columns=self._col_names), n=name)
            self._type = None
            self._values.loc[:, self._col_names[2]] = self._values[self._col_names[2]].astype('category')
        elif is_data(sequence):
            super(Numeric, self).__init__(v=sequence.values, n=name)
            self._type = sequence.data_type
            self._auto_groups = sequence.auto_groups
        else:
            sequence = pd.to_numeric(self.data_prep(sequence), errors='coerce')
            other = pd.to_numeric(self.data_prep(other), errors='coerce') if other is not None else np.nan
            groups = self.data_prep(groups) if groups is not None else 1
            # TODO: This try block needs some work
            try:
                self._values = pd.DataFrame([])
                self._values[self._col_names[0]] = sequence
                self._values[self._col_names[1]] = other
                self._values[self._col_names[2]] = groups
                self._values.loc[:, self._col_names[2]] = self._values[self._col_names[2]].astype('category')
            except ValueError:
                raise UnequalVectorLengthError('length of data does not match length of other.')
            if any(self._values[self._col_names[1]].notnull()):
                self._values = self.drop_nan_intersect()
            else:
                start_time = datetime.datetime.now()
                self._values = self.drop_nan()
                end_time = datetime.datetime.now()
            self._type = self._values[self._col_names[0]].dtype
            self._name = name

    @staticmethod
    def data_prep(seq):
        """
        Converts the values of _name to conform to the Data object standards.

        Parameters
        ----------
        seq : array-like
            The input array to be prepared.

        Returns
        -------
        data : np.array
            The enclosed data represented as a numpy array.
        """
        if hasattr(seq, 'shape'):
            if len(seq.shape) > 1:
                return flatten(seq)
            else:
                return seq
        else:
            return flatten(seq)

    def drop_nan(self):
        """
        Removes NaN values from the Numeric object and returns the resulting pandas Series. The length of the returned
        object is the original object length minus the number of NaN values removed from the object.

        Returns
        -------
        arr : pandas.DataFrame
            A copy of the Numeric object's internal Series with all NaN values removed.
        """
        return self._values.dropna(how='any', subset=[self._col_names[0]])

    def drop_nan_intersect(self):
        """
        Removes the value from the internal Vector object and seq at i where i is nan in the internal Vector object or
        seq.

        Returns
        -------
        arr : pandas.DataFrame
            A tuple of numpy Arrays corresponding to the internal Vector and seq with all nan values removed.
        """
        return self._values.dropna(how='any', subset=[self._col_names[0], self._col_names[1]]) # .reset_index(drop=True)

    @property
    def data_type(self):
        return self._type

    @property
    def data(self):
        return self._values[self._col_names[0]]

    @property
    def other(self):
        return pd.Series([]) if all(self._values[self._col_names[1]].isnull()) else self._values[self._col_names[1]]

    @property
    def groups(self):
        return {grp: seq[self._col_names[0]].rename(grp)
                for grp, seq in self._values.groupby(self._col_names[2])
                if not seq.empty}

    @property
    def paired_groups(self):
        return {grp: (df[self._col_names[0]], df[self._col_names[1]])
                for grp, df in self._values.groupby(self._col_names[2]) if not df.empty}

    @property
    def values(self):
        return self._values

    @property
    def auto_groups(self):
        return self._auto_groups


class Vector(Numeric):
    """
    The sci_analysis representation of continuous, numeric data.
    """

    def __init__(self, sequence=None, other=None, groups=None, name=None):
        """
        Takes an array-like object and converts it to a pandas Series of
        dtype float64, with any non-numeric values converted to NaN.

        Parameters
        ----------
        sequence : array-like or int or float or None
            The input object
        other : array-like
            The secondary input object
        groups : array-like
            The sequence of group names for sub-arrays
        name : str, optional
            The name of the Vector object
        """

        super(Vector, self).__init__(sequence=sequence, other=other, groups=groups, name=name)
        if not self._values.empty:
            self._values[self._col_names[0]] = self._values[self._col_names[0]].astype('float')
            self._values[self._col_names[1]] = self._values[self._col_names[1]].astype('float')

    def is_empty(self):
        """
        Overrides the super class's method to also check for length of zero.

        Returns
        -------
        test_result : bool
            The result of whether the length of the Vector object is 0 or not.

        Examples
        --------
        >>> Vector([1, 2, 3, 4, 5]).is_empty()
        False

        >>> Vector([]).is_empty()
        True
        """
        return self._values.empty

    def append(self, other):
        """
        Append the values of another vector to self.

        Parameters
        ----------
        other : Vector
            The Vector object to be appended to self.

        Returns
        -------
        vector : Vector
            The original Vector object with new values.

        Examples
        --------
        >>> Vector([1, 2, 3]).append(Vector([4, 5, 6])).data
        pandas.Series([1., 2., 3., 4., 5., 6.])
        """
        if not is_vector(other):
            raise ValueError("Vector object cannot be added to a non-vector object.")
        if other.data.empty:
            return self
        if self.auto_groups and other.auto_groups and len(self._values) > 0:
            new_cat = max(self._values[self._col_names[2]].cat.categories) + 1
            other.values['grp'] = new_cat
        self._values = pd.concat([self._values, other.values], copy=False)
        self._values.reset_index(inplace=True, drop=True)
        self._values.loc[:, self._col_names[2]] = self._values[self._col_names[2]].astype('category')
        return self

    def flatten(self):
        """
        Disassociates independent and dependent data into individual groups.

        Returns
        -------
        data : tuple(Series)
            A tuple of pandas Series.
        """
        if not self.other.empty:
            return (tuple(data[self._col_names[0]] for grp, data in self.values.groupby(self._col_names[2])) +
                    tuple(data[self._col_names[1]] for grp, data in self.values.groupby(self._col_names[2])))
        else:
            return tuple(data[self._col_names[0]] for grp, data in self.values.groupby(self._col_names[2]))
