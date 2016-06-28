"""sci_analysis module: vector
Classes:
    Vector - The base data container class used by sci_analysis
"""
from __future__ import absolute_import
# Import from numpy
import numpy as np

# Import from local
from .data import Data


class Vector(Data):
    """The base data container class used by sci_analysis."""

    data_type = "Vector"

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
    def type(self):
        return self._type

    def append_vector(self, vector):
        """Appends sequence from vector to this Vector.

        :param vector: A numPy Array or Vector object
        :return: A copy of the new Vector
        """
        if is_array(vector):
            return np.append(self._values, vector)
        elif is_vector(vector):
            return np.append(self._values, vector.data)

    def is_empty(self):
        """Overrides the super class's method to also check for length of zero.

        :return: True or False
        """
        # return True if self._values is None or len(self._values) == 0 else False
        # return True if self._values is None else False
        return True if len(self._values) == 0 else False

# Perform the operations import at the end to avoid cyclical imports.
from ..operations.data_operations import is_iterable, is_array, is_dict, to_float, is_vector, flatten
