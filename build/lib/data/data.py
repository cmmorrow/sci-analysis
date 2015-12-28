class Data(object):
    """The super class used by all objects representing data for analysis
    in sci_analysis. All analysis classes should expect the data provided through
    arguments to be a descendant of this class.

    Data members are data_type, data and name. data_type is used for identifying
    the container class. The data member stores the data provided through an
    argument. The name member is an optional name for the Data object.
    """

    data_type = "Data"

    def __init__(self, d=None, n=None):
        """Sets the data and name members."""
        self.data = d
        self.name = n

    def is_empty(self):
        """Tests if this Data object's data member equals 'None' and returns
        the result."""
        if self.data is None:
            return True
        else:
            return False

    def __repr__(self):
        """Prints the Data object using the same representation as its data member"""
        return self.data.__repr__()

    def __len__(self):
        """Returns the length of the data member. If data is not defined, 0 is
        returned. If the data member is a scalar value, 1 is returned."""
        if self.data is not None:
            try:
                l = len(self.data)
                return l
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
            return self.data[item]
        except (IndexError, AttributeError):
            return None

    def __contains__(self, item):
        try:
            return item in self.data
        except AttributeError:
            return None

    def __iter__(self):
        """Give this Data object the iterative behavior of its data member."""
        try:
            return self.data.__iter__()
        except AttributeError:
            return None
