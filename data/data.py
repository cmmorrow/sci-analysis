

class Data(object):
    """The data container super class used by sci_analysis"""

    data_type = "Data"

    def __init__(self, d=None, n=None):
        self.data = d
        self.name = n

    def is_empty(self):
        """Tests if a Data object's data member equals 'None'"""
        if self.data is None:
            return True
        else:
            return False

    def __repr__(self):
        return self.data.__repr__()

    def __len__(self):
        if self.data is not None:
            try:
                l = len(self.data)
                return l
            except TypeError:
                return 1
        else:
            return 0

    def __getitem__(self, item):
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
        try:
            return self.data.__iter__()
        except AttributeError:
            return None
