"""sci_analysis module: data
Functions:
    is_vector: test if the passed array_like argument is a sci_analysis Vector object.
    is_data: test if the passed array_like argument is a sci_analysis Data object.
"""


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
