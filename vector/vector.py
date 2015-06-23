import numpy as np
from operations import is_iterable
from operations import is_array
from operations import is_dict


class Vector:
    """ The base data container class used by sci-analysis
    """

    def __init__(self, data):

        self.data = np.empty
        if is_array(data):
            self.data = data
        else:
            if is_dict(data):
                data = self.flatten(data.values())
            if is_iterable(data):
                self.data = np.array(self.to_float(data))
        self.__len__ = len(self.data)
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

    def append_to(self, vector):
        """ Appends data from vector to this Vector
        """
        np.append(self.data, vector.data)

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
