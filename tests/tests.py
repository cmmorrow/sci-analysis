import scipy.stats as st
from ..vector import vector
from ..vector import operations


class Test:
    """ Generic statistical test class
    """

    def __init__(self, data, alpha=0.05, display=True):
        self.data = data
        self.alpha = alpha
        self.display = display
        self.results = 0, 1
        self.run()
        if display:
            self.output()
            if self.results[1] > alpha:
                self.h0()
            else:
                self.ha()

    def run(self):
        self.results = results = 0, 1
        return results

    def output(self):
        values = self.run()
        print str(values[0]) + ", " + str(values[1])

    def h0(self):
        print "H0: "
        pass

    def ha(self):
        print "HA: "
        pass


class NormTest(Test):
    """ Tests for whether data is normally distributed or not
    """

    def run(self):



        if not operations.is_vector(self.data):
            data = operations.cat(self.data)
        else:
            data = operations.cat(self.data.data)
        data = vector.Vector(self.data)



