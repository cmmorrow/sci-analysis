# Import from numpy
import numpy as np

# Import from local
from data import Data
from operations import is_iterable, is_array, is_dict


class Vector(Data):
    """ The base data container class used by sci-analysis
    """

    data_type = "Vector"

    def __init__(self, data=None, name=None):

        super(Vector, self).__init__(None, name)
        if is_array(data):
            self.data = data
        else:
            if is_dict(data):
                data = self.flatten(data.values())
            if is_iterable(data):
                self.data = np.array(self.to_float(data))
                self.type = self.data.dtype

    def to_float(self, data):
        """ Converts values in data to float and returns a copy
        """
        float_list = []
        for i in range(len(data)):
            try:
                float_list.append(float(data[i]))
            except ValueError:
                float_list.append(float("nan"))
        return float_list

    def append_vector(self, vector):
        """ Appends data from vector to this Vector
        """
        return np.append(self.data, vector.data)

    def flatten(self, data):
        """ Reduce the dimension of data by one
        """
        flat = []
        for row in data:
            if is_iterable(row):
                for col in row:
                    flat.append(col)
            else:
                flat.append(row)
        return flat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __contains__(self, item):
        return item in self.data
