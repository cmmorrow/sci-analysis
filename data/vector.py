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

    def __init__(self, data=np.array([]), name=None):
        """Takes a sequence like object and converts it to a numPy Array of
        dtype float64, with any non-numeric values converted to NaN.

        :param data: A numPy Array of data type float64
        :param name: An optional name for the Vector object
        :return: A Vector object
        """

        super(Vector, self).__init__(n=name)
        if is_array(data):
            try:
                self.data = np.asfarray(data)
                self.type = self.data.dtype
            except ValueError:
                self.data = np.array([])
        elif is_vector(data):
            self.data = data.data
            self.name = data.name
            self.type = data.type
        else:
            if is_dict(data):
                data = flatten(list(data.values()))
            if is_iterable(data):
                self.data = np.array(to_float(data))
                self.type = self.data.dtype
            else:
                self.data = np.array([])
        if len(self.data.shape) > 1:
            self.data = self.data.flatten()

    def append_vector(self, vector):
        """Appends data from vector to this Vector.

        :param vector: A numPy Array or Vector object
        :return: A copy of the new Vector
        """
        if is_array(vector):
            return np.append(self.data, vector)
        elif is_vector(vector):
            return np.append(self.data, vector.data)

    def is_empty(self):
        """Overrides the super class's method to also check for length of zero.

        :return: True or False
        """
        if self.data is None or len(self.data) == 0:
            return True
        else:
            return False

# Perform the operations import at the end to avoid cyclical imports.
from ..operations.data_operations import is_iterable, is_array, is_dict, to_float, is_vector, flatten
