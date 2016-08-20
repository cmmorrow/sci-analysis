from __future__ import absolute_import
# Import from numpy
import numpy as np

# Import from local
from ..operations.data_operations import is_array, is_dict, is_iterable, to_float, flatten


def assign(obj, other=None):
    return (Vector(obj), Vector(other)) if other is not None else Vector(obj)


def is_vector(obj):
    """Checks if the argument is a Vector object.

    :param obj: A variable of unknown type
    :return: True or False
    """
    return isinstance(obj, Vector)


def is_data(obj):
    """Checks if the argument is a Data object.

    :param obj: A variable of unknown type
    :return: True or False
    """
    if isinstance(obj, Data):
        return True
    else:
        return False


class EmptyVectorError(Exception):
    pass


class UnequalVectorLengthError(Exception):
    pass


class Data(object):
    """The super class used by all objects representing data for analysis
    in sci_analysis. All analysis classes should expect the data provided through
    arguments to be a descendant of this class.

    Data members are data_type, data and name. data_type is used for identifying
    the container class. The data member stores the data provided through an
    argument. The name member is an optional name for the Data object.
    """

    def __init__(self, v=None, n=None):
        """Sets the data and name members."""
        self._values = v
        self._name = n

    def data_prep(self):
        return self.data

    def is_empty(self):
        """Tests if this Data object's data member equals 'None' and returns
        the result."""
        return self._values is None

    @property
    def data(self):
        return self._values

    @property
    def name(self):
        return self._name

    def __repr__(self):
        """Prints the Data object using the same representation as its data member"""
        return self._values.__repr__()

    def __len__(self):
        """Returns the length of the data member. If data is not defined, 0 is
        returned. If the data member is a scalar value, 1 is returned."""
        if self._values is not None:
            try:
                return len(self._values)
            except TypeError:
                return 1
        else:
            return 0

    def __getitem__(self, item):
        """Gets the value of the data member at index item and returns it.

        :param item: An index of the data member
        :return: Returns the value of the data member at the index specified
        by item, or returns None if no such index exists
        """
        try:
            return self._values[item]
        except (IndexError, AttributeError):
            return None

    def __contains__(self, item):
        try:
            return item in self._values
        except AttributeError:
            return None

    def __iter__(self):
        """Give this Data object the iterative behavior of its data member."""
        try:
            return self._values.__iter__()
        except AttributeError:
            return None


class Vector(Data):
    """The base data container class used by sci_analysis."""

    def __init__(self, sequence=None, name=None):
        """Takes a sequence like object and converts it to a numPy Array of
        dtype float64, with any non-numeric values converted to NaN.

        :param sequence: A numPy Array of data type float64
        :param name: An optional name for the Vector object
        :return: A Vector object
        """

        super(Vector, self).__init__(v=sequence, n=name)
        if is_vector(sequence):
            self._values = sequence.data
            self._name = sequence.name
            self._type = sequence.data_type
        elif is_array(sequence):
            try:
                self._values = np.asfarray(sequence)
                self._type = self._values.dtype
            except ValueError:
                self._values = np.array(to_float(sequence))
                self._type = self._values.dtype
        else:
            if is_dict(sequence):
                values = list(sequence.values())
                sequence = flatten(values)
            if is_iterable(sequence):
                self._values = np.array(to_float(sequence))
                self._type = self._values.dtype
            else:
                try:
                    self._values = np.array([float(sequence)])
                except (ValueError, TypeError):
                    self._values = np.array([])
        if len(self._values.shape) > 1:
            self._values = self._values.flatten()

    @property
    def data_type(self):
        return self._type

    def data_prep(self, other=None):
        """

        :param other:
        :return:
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
        """Overrides the super class's method to also check for length of zero.

        :return: True or False
        """
        return len(self._values) == 0

    def drop_nan(self):
        """ Removes NaN values from the given sequence v and returns the equivalent
        Vector object. The length of the returned Vector is the length of v minus
        the number of "nan" values removed from v.

        :return: A vector representation of v with "nan" values removed
        """
        return np.array([]) if self.is_empty() else self.data[~np.isnan(self.data)]

    def drop_nan_intersect(self, seq):
        """Takes two sequence like arguments first and second, and creates a
        tuple of sequences where only non-NaN values are given. This is accomplished
        by removing values that are "nan" on matching indicies of either first or second.

        :param seq: A sequence like object
        :return: A two element tuple of Vector objects with "nan" values removed
        """
        if self.is_empty() or seq.is_empty():
            return np.array([]), np.array([])
        c = np.logical_and(~np.isnan(self.data), ~np.isnan(seq.data))
        if not any(c):
            return np.array([]), np.array([])
        return self.data[c], seq.data[c]
