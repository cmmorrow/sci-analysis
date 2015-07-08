

class Data(object):

    data_type = "Data"

    def __init__(self, d=None, n=None):
        self.data = d
        self.name = n

    def is_empty(self):
        if self.data is None:
            return True
        else:
            return False
