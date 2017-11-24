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
        print('starting')
        self._auto_groups = True if groups is None else False
        if sequence is None:
            super(Numeric, self).__init__(v=pd.DataFrame([], columns=self._col_names), n=name)
            self._type = None
            # self._auto_groups = False
            self._values.loc[:, self._col_names[2]] = self._values[self._col_names[2]].astype('category')
        elif is_data(sequence):
            super(Numeric, self).__init__(v=sequence.values, n=name)
            self._type = sequence.data_type
            self._auto_groups = sequence.auto_groups
        # elif not is_iterable(sequence):
        #     super(Numeric, self).__init__(v=pd.DataFrame([sequence], columns=self._col_names), n=name)
        #     self._type = type(sequence)
        # elif is_iterable(sequence):
            # if hasattr(sequence, 'shape'):
            #     if len(sequence.shape) > 1:
            #         sequence = sequence.flatten()
            # elif not is_dict(sequence):
            #     sequence = flatten(sequence)
            # self._values = pd.to_numeric(pd.Series(sequence), errors='coerce')
            # self._type = self._values.dtype
        else:
            sequence = pd.to_numeric(self.data_prep(sequence), errors='coerce')
            other = pd.to_numeric(self.data_prep(other), errors='coerce') if other is not None else np.nan
            groups = self.data_prep(groups) if groups is not None else 1
            # TODO: This try block needs some work
            try:
                # self._values = pd.to_numeric(pd.Series([sequence], index=[0]), errors='coerce')
                print('dataframe creation')
                start_time = datetime.datetime.now()
                self._values = pd.DataFrame([])
                self._values[self._col_names[0]] = sequence
                self._values[self._col_names[1]] = other
                self._values[self._col_names[2]] = groups
                self._values.loc[:, self._col_names[2]] = self._values[self._col_names[2]].astype('category')
                # print(self._values)
                end_time = datetime.datetime.now()
                print(end_time - start_time)
            # except (ValueError, TypeError):
            except ValueError:
                # self._values = pd.DataFrame([], columns=self._col_names)
                raise UnequalVectorLengthError('length of data does not match length of other.')
            if any(self._values[self._col_names[1]].notnull()):
                self._values = self.drop_nan_intersect()
            else:
                print('starting drop_nan')
                start_time = datetime.datetime.now()
                self._values = self.drop_nan()
                end_time = datetime.datetime.now()
                print(end_time - start_time)
            print(self._values)
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
        # if is_dict(seq):
        #     return [v for v in seq.values()]
        print('preparing')
        start_time = datetime.datetime.now()
        if hasattr(seq, 'shape'):
            if len(seq.shape) > 1:
                return flatten(seq)
            else:
                end_time = datetime.datetime.now()
                print(end_time - start_time)
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
        # return self._values.dropna().reset_index(drop=True)
        return self._values.dropna(how='any', subset=[self._col_names[0]]) # .reset_index(drop=True)

    def drop_nan_intersect(self):
        """
        Removes the value from the internal Vector object and seq at i where i is nan in the internal Vector object or
        seq.

        Returns
        -------
        arr : pandas.DataFrame
            A tuple of numpy Arrays corresponding to the internal Vector and seq with all nan values removed.
        """
        # c = pd.DataFrame({'self': self._values, 'other': seq}).dropna().reset_index(drop=True)
        # return c['self'], c['other']
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
            # if self._values.dtype != self._type:
            #     self._type = self._values.dtype

    # def data_prep(self, other=None):
    #     """
    #     Remove all nan values from the encapsulated Array.
    #
    #     Parameters
    #     ----------
    #     other : array-like, optional
    #         A secondary array-like object with corresponding NaN values to remove.
    #
    #     Returns
    #     -------
    #     vector : Vector
    #         A vector object with all nan values removed.
    #     """
    #     if other is not None:
    #         vector = other if is_vector(other) else Vector(other)
    #         if len(self.data) != len(vector.data):
    #             raise UnequalVectorLengthError("x and y data lengths are not equal")
    #
    #         x, y = self.drop_nan_intersect(vector)
    #
    #         if len(x) == 0 or len(y) == 0:
    #             return None
    #         return x, y
    #     elif not is_iterable(self.data):
    #         return pd.Series(self.data).astype(float)
    #     else:
    #         v = self.drop_nan()
    #         if len(v) == 0:
    #             return None
    #         return v

    def is_empty(self):
        """
        Overrides the super class's method to also check for length of zero.

        Returns
        -------
        test_result : bool
            The result of whether the length of the Vector object is 0 or not.
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
