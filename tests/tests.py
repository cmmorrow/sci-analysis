# Scipy imports
from scipy.stats import linregress, shapiro, pearsonr, spearmanr, f_oneway, kruskal, bartlett, levene

# Numpy imports
from numpy import concatenate

# Local imports
from ..data import vector, operations


class Test(object):
    """ Generic statistical test class
    """

    def __init__(self, data, alpha=0.05, display=True):

        # Set members
        self.data = data
        self.alpha = alpha
        self.display = display
        self.results = 1, 0

        # If data is not a vector, wrap it in a Vector object
        if not operations.is_vector(data):
            self.data = vector.Vector(data)

        # Stop the test if the vector is empty
        if self.data.is_empty():
            print "vector is empty"
            pass
        else:

            # Remove NaN values from the vector
            self.data = operations.drop_nan(self.data)

            # Run the test and display the results
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
        return 1, 0

    def output(self):
        print str(self.results[1]) + ", " + str(self.results[0])

    def h0(self):
        print "H0: "

    def ha(self):
        print "HA: "


class GroupTest(Test):

    def __init__(self, *groups, **parms):
        self.alpha = 0.05
        self.data = []
        self.display = True
        self.results = 1, 0

        parm_list = sorted(parms.keys())
        for parm in parm_list:
            if parm == "alpha":
                self.alpha = parm
            if parm == "display":
                self.display = parm
        if operations.is_dict(groups[0]):
            groups = groups[0].values()
        for group in groups:
#            if len(group) == 1:
#                continue
            if not operations.is_vector(group):
                group = vector.Vector(group)
            if group.is_empty():
                continue
            if len(group) == 1:
                continue
            self.data.append(group.data)
        super(GroupTest, self).logic()


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
        if len(xdata) != len(ydata):
            print "Vector lengths are not equal"
            pass
        elif xdata.is_empty() or ydata.is_empty():
            print "At least one vector is empty"
            pass
        else:

            self.xdata, self.ydata = operations.drop_nan_intersect(self.xdata, self.ydata)

    def logic(self):
        if len(self.xdata) <= self.__min_size or len(self.ydata) <= self.__min_size:
            return self.results
        super(Comparison, self).logic()


class NormTest(Test):
    """ Tests for whether data is normally distributed or not
    """

    def run(self):
        w_value, p_value = shapiro(self.data)
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
        slope, intercept, r2, p_value, std_err = linregress(self.xdata, self.ydata)
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
        if NormTest(concatenate([self.xdata, self.ydata]), display=False, alpha=self.alpha).results[0] > self.alpha:
            r_value, p_value = pearsonr(self.xdata, self.ydata)
            r = "pearson"
        else:
            r_value, p_value = spearmanr(self.xdata, self.ydata)
            r = "spearman"
        return p_value, r_value, r

    def output(self):
        name = "Correlation"
        print ""
        print name
        print "-" * len(name)
        print ""
        if self.results[2] == "pearson":
            print "Pearson Coeff:"
        else:
            print "Spearman Coeff:"
        print "r = " + "{:.4f}".format(self.results[1])
        print "p = " + "{:.4f}".format(self.results[0])
        print ""

    def h0(self):
        print "H0: There is no significant relationship between predictor and response"

    def ha(self):
        print "HA: There is a significant relationship between predictor and response"


class Anova(GroupTest):

    __min_size = 2

    def run(self):
        if len(self.data) <= self.__min_size:
            return self.results
        f_value, p_value = f_oneway(*tuple(self.data))
        return p_value, f_value

    def output(self):
        name = "Oneway ANOVA"
        print ""
        print name
        print "-" * len(name)
        print ""
        print "f value = " + "{:.4f}".format(self.results[1])
        print "p value = " + "{:.4f}".format(self.results[0])
        print ""

    def h0(self):
        print "H0: Group means are matched"

    def ha(self):
        print "HA: Group means are not matched"


class Kruskal(GroupTest):

    __min_size = 2

    def run(self):
        if len(self.data) <= self.__min_size:
            return self.results
        h_value, p_value = kruskal(*tuple(self.data))
        return p_value, h_value

    def output(self):
        name = "Kruskal-Wallis"
        print ""
        print name
        print "-" * len(name)
        print ""
        print "H value = " + "{:.4f}".format(self.results[1])
        print "p value = " + "{:.4f}".format(self.results[0])
        print ""

    def h0(self):
        print "H0: Group means are matched"

    def ha(self):
        print "HA: Group means are not matched"


class EqualVariance(GroupTest):

    __min_size = 2

    def run(self):
        if len(self.data) <= self.__min_size:
            return self.results
        if NormTest(concatenate(self.data), display=False, alpha=self.alpha).results[0] > self.alpha:
            statistic, p_value = bartlett(*tuple(self.data))
            t = "Bartlett Test"
        else:
            statistic, p_value = levene(*tuple(self.data))
            t = "Levene Test"
        return p_value, statistic, t

    def output(self):
        name = "Equal Variance"
        print ""
        print name
        print "-" * len(name)
        print ""
        print self.results[2]
        if self.results[2] == "Bartlett Test":
            print "T value = " + "{:.4f}".format(self.results[1])
        else:
            print "W value = " + "{:.4f}".format(self.results[1])
        print "p value = " + "{:.4f}".format(self.results[0])

    def h0(self):
        print "H0: Variances are equal"

    def ha(self):
        print "HA: Variances are not equal"
