import numpy as np

class Vector:
    """ The base data container class used by sci-analysis
    """
    __min_size = 0

    def __init__(self, data):
        if self.isarray(data):
            self.data = data
        else:
            if self.isdict(data):
                data = self.flatten(data)
            if self.isiterable(data):
                self.data = np.array(self.to_float(data))
            else:
                self.data = np.empty

    def isarray(self, data):
        """ Tests if data is a numPy Array object
        """
        try:
            data.shape
            return True
        except AttributeError:
            return False

    def isdict(self, data):
        """ Test if data is a dictionary object
        """
        try:
            data.keys()
            return True
        except AttributeError:
            return False

    def isiterable(self, data):
        """ Tests if data is a collection and not empty
        """
        try:
            if len(data) > self.__min_size:
                return True
            else:
                return False
        except TypeError:
            return False

    def to_float(self, data):
        """ Converts values in data to float
        """
        for i in range(len(data)):
            try:
                data[i] = float(data[i])
            except ValueError:
                data[i] = float("nan")
        return data

    def append(self, vector):
        """ Appends data from vector to this Vector
        """
        np.append(self.data, vector.data)

    def flatten(self, data):
        return [[data.append(col) for col in row] for row in data.values()]