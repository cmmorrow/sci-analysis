# Import from numpy
import numpy as np

# Import from local
from data import Data
from operations import is_iterable, is_array, is_dict, to_float, is_vector, flatten


class Vector(Data):
    """The base data container class used by sci-analysis"""

    data_type = "Vector"

    def __init__(self, data=np.array([]), name=None):

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
                data = flatten(data.values())
            if is_iterable(data):
                self.data = np.array(to_float(data))
                self.type = self.data.dtype
            else:
                self.data = np.array([])
        if len(self.data.shape) > 1:
            self.data = self.data.flatten()

    def append_vector(self, vector):
        """Appends data from vector to this Vector"""
        if is_array(vector):
            return np.append(self.data, vector)
        elif is_vector(vector):
            return np.append(self.data, vector.data)

    def is_empty(self):
        """Over-rides the super class's method to also check for len = 0"""
        if self.data is None or len(self.data) == 0:
            return True
        else:
            return False
