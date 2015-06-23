import scipy.stats as st
from ..vector import vector
from ..vector import operations


class Test:
    """ Generic statistical test class
    """

    def __init__(self, data, alpha=0.05, display=True):

        """Set members """
        self.data = data
        self.alpha = alpha
        self.display = display
        self.results = 0, 1

        """If data is not a vector, wrap it in a vector object """
        if not operations.is_vector(data):
            self.data = vector.Vector(data)

        """Remove NaN values from the vector"""
        self.data = operations.drop_nan(self.data)

        """Run the test and display the results"""
        self.results = self.run()
        if display:
            self.output()
            if self.results[1] > alpha:
                self.h0()
            else:
                self.ha()
            print ""

    def run(self):
        return 0, 1

    def output(self):
        print str(self.results[0]) + ", " + str(self.results[1])

    def h0(self):
        print "H0: "

    def ha(self):
        print "HA: "


class NormTest(Test):
    """ Tests for whether data is normally distributed or not
    """

    def run(self):
        w_value, p_value = st.shapiro(self.data)
        return w_value, p_value

    def output(self):
        name = "Shapiro-Wilk test for normality"
        print ""
        print name
        print "-" * len(name)
        print ""
        print "W value = " + "{:.4f}".format(self.results[0])
        print "p value = " + "{:.4f}".format(self.results[1])

    def h0(self):
        print "H0: Data is normally distributed"

    def ha(self):
        print "HA: Data is not normally distributed"
