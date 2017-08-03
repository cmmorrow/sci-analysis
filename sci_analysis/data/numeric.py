# Import packages
import pandas as pd

# Import from local
from .data import Data, is_data
from .data_operations import is_iterable, is_dict, flatten


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
        elif is_data(sequence):
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
            return pd.Series(self.data).astype(float)
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
        return self._values.empty
