"""sci_analysis module: data
Functions:
    assign: convert the passed array_like arguments to the correct sci_analysis object.
    is_vector: test if the passed array_like argument is a sci_analysis Vector object.
    is_data: test if the passed array_like argument is a sci_analysis Data object.
"""
from __future__ import absolute_import
# Import from numpy
import numpy as np

# Import from local
from ..operations.data_operations import is_array, is_dict, is_iterable, to_float, flatten


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

    def data_prep(self):
        """
        Converts the values of _name to conform to the Data object standards.

        Returns
        -------
        data : np.array
            The enclosed data represented as a numpy array.
        """
        return self.data

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


class Vector(Data):
    """
    The sci_analysis representation of continuous, numeric data.
    """

    def __init__(self, sequence=None, name=None):
        """
        Takes an array-like object and converts it to a numpy Array of
        dtype float64, with any non-numeric values converted to nan.

        Parameters
        ----------
        sequence : array-like
            The input object
        name : str, optional
            The name of the Vector object
        """

        super(Vector, self).__init__(v=sequence, n=name)
        if is_vector(sequence):
            # Create a copy of the input Vector
            self._values = sequence.data
            self._name = sequence.name
            self._type = sequence.data_type
        elif is_array(sequence):
            # Convert the Array dtype to float64
            try:
                self._values = np.asfarray(sequence)
                self._type = self._values.dtype
            # Convert each value of the Array to a float or nan (which is technically a float)
            except ValueError:
                self._values = np.array(to_float(sequence))
                self._type = self._values.dtype
        else:
            # Convert the input dict to an Array
            if is_dict(sequence):
                values = list(sequence.values())
                sequence = flatten(values)
            # Convert the python list or tuple to an Array
            if is_iterable(sequence):
                self._values = np.array(to_float(sequence))
                self._type = self._values.dtype
            else:
                # Convert a single value to a 1d Array
                try:
                    self._values = np.array([float(sequence)])
                # Create an empty Array
                except (ValueError, TypeError):
                    self._values = np.array([])
        # Flatten a multi-dimensional Array
        if len(self._values.shape) > 1:
            self._values = self._values.flatten()

    @property
    def data_type(self):
        return self._type

    def data_prep(self, other=None):
        """
        Remove all nan values from the encapsulated Array.

        Parameters
        ----------
        other : array-like, optional
            A secondary array-like object with corresponding nan values to remove.

        Returns
        -------
        vector : Vector
            A vector object with all nan values removed.
        """
        if other is not None:
            vector = other if is_vector(other) else Vector(other)
            if len(self.data) != len(vector):
                raise UnequalVectorLengthError("x and y data lengths are not equal")

            x, y = self.drop_nan_intersect(vector)

            if len(x) == 0 or len(y) == 0:
                return None
                # raise EmptyVectorError("Passed data is empty")
            return x, y
        elif not is_iterable(self.data):
            return np.array(float(self.data))
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
        return len(self._values) == 0

    def drop_nan(self):
        """
        Removes nan values from the Vector object and returns a numpy Array. The length of the returned Vector is the
        length of the Vector object minus the number of nan values removed from the Vector object.

        Returns
        -------
        arr : numpy.Array
            A copy of the Vector object's internal Array with all nan values removed.
        """
        return np.array([]) if self.is_empty() else self.data[~np.isnan(self.data)]

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
        if self.is_empty() or seq.is_empty():
            return np.array([]), np.array([])
        c = np.logical_and(~np.isnan(self.data), ~np.isnan(seq.data))
        if not any(c):
            return np.array([]), np.array([])
        return self.data[c], seq.data[c]
