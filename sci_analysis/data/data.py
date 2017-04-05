"""sci_analysis module: data
Functions:
    assign: convert the passed array_like arguments to the correct sci_analysis object.
    is_vector: test if the passed array_like argument is a sci_analysis Vector object.
    is_data: test if the passed array_like argument is a sci_analysis Data object.
"""
from __future__ import absolute_import
# Import packages
import pandas as pd

# Import from local
from ..operations.data_operations import is_dict, is_iterable, flatten


def assign(obj, other=None):
    """
    Convert the passed array_like arguements to the correct sci_analysis object.

    Parameters
    ----------
    obj : object
        The input object.
    other : object of unknown type, optional
        A secondary object corresponding to the input object

    Returns
    -------
    subseq : Data
        A sci_analysis object that inherits from Data
    """
    return (Vector(obj), Vector(other)) if other is not None else Vector(obj)


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
        elif isinstance(sequence, Data):
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
        # return np.array([]) if self.is_empty() else self.data[~np.isnan(self.data)]
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
        # if self.is_empty() or seq.is_empty():
        #     return np.array([]), np.array([])
        # c = np.logical_and(~np.isnan(self.data), ~np.isnan(seq.data))
        # if not any(c):
        #     return np.array([]), np.array([])
        # return self.data[c], seq.data[c]
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
        # self._values = None
        # self._type = None
        # self._name = name or None
        # if is_vector(sequence):
        #     # Create a copy of the input Vector
        #     self._values = sequence.data
        #     self._name = sequence.name
        #     self._type = sequence.data_type
        # elif is_iterable(sequence):
        #     if hasattr(sequence, 'shape'):
        #         if len(sequence.shape) > 1:
        #             sequence = sequence.flatten()
        #     elif not is_dict(sequence):
        #         sequence = flatten(sequence)
        #     self._values = pd.to_numeric(pd.Series(sequence), errors='coerce').astype('float')
        #     self._type = self._values.dtype
        # else:
        #     try:
        #         self._values = pd.Series([float(sequence)], index=[0])
        #     except (ValueError, TypeError):
        #         self._values = pd.Series([])


        # elif is_array(sequence):
        #     # Convert the Array dtype to float64
        #     try:
        #         self._values = np.asfarray(sequence)
        #         self._type = self._values.dtype
        #     # Convert each value of the Array to a float or nan (which is technically a float)
        #     except ValueError:
        #         self._values = np.array(to_float(sequence))
        #         self._type = self._values.dtype
        # else:
        #     # Convert the input dict to an Array
        #     if is_dict(sequence):
        #         values = list(sequence.values())
        #         sequence = flatten(values)
        #     # Convert the python list or tuple to an Array
        #     if is_iterable(sequence):
        #         self._values = np.array(to_float(sequence))
        #         self._type = self._values.dtype
        #     else:
        #         # Convert a single value to a 1d Array
        #         try:
        #             self._values = np.array([float(sequence)])
        #         # Create an empty Array
        #         except (ValueError, TypeError):
        #             self._values = np.array([])
        # Flatten a multi-dimensional Array
        # if len(self._values.shape) > 1:
        #     self._values = self._values.flatten()

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
            return pd.Series(float(self.data))
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

