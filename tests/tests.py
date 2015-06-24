import scipy.stats as st
import numpy as np
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
        self.results = 1, 0

        """If data is not a vector, wrap it in a vector object """
        if not operations.is_vector(data):
            self.data = vector.Vector(data)

        """Remove NaN values from the vector"""
        self.data = operations.drop_nan(self.data)

        """Run the test and display the results"""
        self.logic()

    def logic(self):
        self.results = self.run()
        if self.display:
            self.output()
            if self.results[0] > self.alpha:
                self.h0()
            else:
                self.ha()
            print ""

    def run(self):
        return 0, 1

    def output(self):
        print str(self.results[1]) + ", " + str(self.results[0])

    def h0(self):
        print "H0: "

    def ha(self):
        print "HA: "


class Comparison(Test):

    __min_size = 2

    def __init__(self, xdata, ydata, alpha=0.05, display=True):

        self.xdata = xdata
        self.ydata = ydata
        self.alpha = alpha
        self.display = display
        self.results = 1, 0

        if not operations.is_vector(xdata):
            self.xdata = vector.Vector(xdata)
        if not operations.is_vector(ydata):
            self.ydata = vector.Vector(ydata)

        self.xdata, self.ydata = operations.drop_nan_intersect(self.xdata, self.ydata)

        if len(self.xdata) <= self.__min_size or len(self.ydata) <= self.__min_size:
            return self.results

        self.logic()


class NormTest(Test):
    """ Tests for whether data is normally distributed or not
    """

    def run(self):
        w_value, p_value = st.shapiro(self.data)
        return p_value, w_value

    def output(self):
        name = "Shapiro-Wilk test for normality"
        print ""
        print name
        print "-" * len(name)
        print ""
        print "W value = " + "{:.4f}".format(self.results[1])
        print "p value = " + "{:.4f}".format(self.results[0])

    def h0(self):
        print "H0: Data is normally distributed"

    def ha(self):
        print "HA: Data is not normally distributed"


class LinearRegression(Comparison):

    __min_size = 3

    def run(self):
        slope, intercept, r2, p_value, std_err = st.linregress(self.xdata, self.ydata)
        return p_value, slope, intercept, r2, std_err

    def output(self):
        name = "Linear Regression"
        print ""
        print name
        print "-" * len(name)
        print ""
        print "slope     = " + "{:.4f}".format(self.results[1])
        print "intercept = " + "{:.4f}".format(self.results[2])
        print "R^2       = " + "{:.4f}".format(self.results[3])
        print "std err   = " + "{:.4f}".format(self.results[4])
        print "p value   = " + "{:.4f}".format(self.results[0])
        print ""

    def h0(self):
        print "H0: There is no significant relationship between predictor and response"

    def ha(self):
        print "HA: There is a significant relationship between predictor and response"


class Correlation(Comparison):

    __min_size = 3

    def run(self):
        if NormTest(np.concatenate([self.xdata, self.ydata]), display=False, alpha=self.alpha)[0] > self.alpha:
            r_value, p_value = st.pearsonr(self.xdata, self.ydata)
            r = "pearson"
        else:
            r_value, p_value = st.spearmanr(self.xdata, self.ydata)
            r = "spearman"
        return p_value, r_value, r

    def output(self):
        name = "Correlation"
        print ""
        print name
        print "-" * len(name)
        print ""
        if self.results[2] == "pearson":
            print "    Pearson Coeff:"
        else:
            print "    Spearman Coeff:"
        print "r = " + "{:.4f}".format(self.results[1])
        print "p = " + "{:.4f}".format(self.results[0])
        print ""

    def h0(self):
        print "H0: There is no significant relationship between predictor and response"

    def ha(self):
        print "HA: There is a significant relationship between predictor and response"
