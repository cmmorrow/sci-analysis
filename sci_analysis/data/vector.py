"""sci_analysis module: vector
Classes:
    Vector - The base data container class used by sci_analysis
"""
from __future__ import absolute_import
# Import from numpy
import numpy as np

# Import from local
from data.data import Data
from operations.data_operations import drop_nan, drop_nan_intersect, is_vector, is_array, is_dict, is_iterable, \
    to_float, flatten


class EmptyVectorError(Exception):
    pass


class UnequalVectorLengthError(Exception):
    pass


class Vector(Data):
    """The base data container class used by sci_analysis."""

    def __init__(self, sequence=None, name=None):
        """Takes a sequence like object and converts it to a numPy Array of
        dtype float64, with any non-numeric values converted to NaN.

        :param sequence: A numPy Array of data type float64
        :param name: An optional name for the Vector object
        :return: A Vector object
        """

        self._prepared = False
        super(Vector, self).__init__(v=sequence, n=name)
        if is_vector(sequence):
            self._values = sequence.data
            self._name = sequence.name
            self._type = sequence.type
        elif is_array(sequence):
            try:
                self._values = np.asfarray(sequence)
                self._type = self._values.dtype
            except ValueError:
                self._values = np.array(to_float(sequence))
                self._type = self._values.dtype
        else:
            if is_dict(sequence):
                sequence = flatten(list(sequence.values()))
            if is_iterable(sequence):
                self._values = np.array(to_float(sequence))
                self._type = self._values.dtype
            else:
                self._values = np.array([])
        if len(self._values.shape) > 1:
            self._values = self._values.flatten()

    @property
    def data_type(self):
        return self._type

    @property
    def prepared(self):
        return self._prepared

    def data_prep(self, other=None):
        """

        :param other:
        :return:
        """
        if self._prepared:
            return self.data
        if other:
            vector = other if is_vector(other) else Vector(other)
            if len(self.data) != len(vector):
                raise UnequalVectorLengthError("x and y data lengths are not equal")
            self._prepared = True
            return drop_nan_intersect(self.data, vector)
        elif not is_iterable(self.data):
            self._prepared = True
            return float(self.data)
        else:
            v = drop_nan(self.data) if is_vector(self.data) else drop_nan(Vector(self.data))
            if not v:
                raise EmptyVectorError("Passed data is empty")
            self._prepared = True
            return v

    def is_empty(self):
        """Overrides the super class's method to also check for length of zero.

        :return: True or False
        """
        return True if len(self._values) == 0 else False

# Perform the operations import at the end to avoid cyclical imports.
# from ..operations.data_operations import is_iterable, is_array, is_dict, to_float, is_vector, flatten
