# Scipy imports
from scipy.stats import linregress, shapiro, pearsonr, spearmanr, f_oneway, kruskal, bartlett, levene, skew, kurtosis

# Numpy imports
from numpy import concatenate, mean, std, median, amin, amax, percentile

# Local imports
from ..data import vector, operations


class Analysis(object):

    def __init__(self, data):
        self.data = data
        self.results = 0.0

    def logic(self):
        self.results = self.run()
        self.output()

    def run(self):
        pass

    def output(self):
        pass


class Test(Analysis):
    """ Generic statistical test class
    """

    def __init__(self, data, alpha=0.05, display=True):

        super(Test, self).__init__(data)

        # Set members
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


class VectorStatistics(Analysis):

    __min_size = 2

    def __init__(self, data, sample=False):
        super(VectorStatistics, self).__init__(data)

        self.results = None
        self.sample = sample

        if not operations.is_vector(data):
            self.data = vector.Vector(data)

        if self.data.is_empty():
            print "vector is empty"
            pass
        elif len(self.data) < self.__min_size:
            pass
        else:
            # Remove NaN values from the vector
            self.data = operations.drop_nan(self.data)
            if len(self.data) < self.__min_size:
                pass
            else:
                self.logic()

    def run(self):
        dof = 0
        if self.sample:
            dof = 1
        count = len(self.data)
        avg = mean(self.data)
        sd = std(self.data, ddof=dof)
        med = median(self.data)
        vmin = amin(self.data)
        vmax = amax(self.data)
        vrange = vmax - vmin
        sk = skew(self.data)
        kurt = kurtosis(self.data)
        q1 = percentile(self.data, 25)
        q3 = percentile(self.data, 75)
        iqr = q3 - q1
        return {"count": count,
                "mean": avg,
                "std": sd,
                "median": med,
                "min": vmin,
                "max": vmax,
                "range": vrange,
                "skew": sk,
                "kurtosis": kurt,
                "q1": q1,
                "q3": q3,
                "iqr": iqr}

    def output(self):
        name = "Statistics"
        print ""
        print name
        print "-" * len(name)
        print ""
        print "Count = " + str(self.results["count"])
        print "Mean = " + str(self.results['mean'])
        print "Standard Deviation = " + str(self.results['std'])
        print "Skewness = " + str(self.results['skew'])
        print "Kurtosis = " + str(self.results['kurtosis'])
        print "Max = " + str(self.results['max'])
        print "75% = " + str(self.results['q3'])
        print "50% = " + str(self.results['median'])
        print "25% = " + str(self.results['q1'])
        print "Min = " + str(self.results['min'])
        print "IQR = " + str(self.results['iqr'])
        print "Range = " + str(self.results['range'])
        print ""


class GroupStatistics(Analysis):

    __min_size = 1

    def __init__(self, data, groups=None):
        super(GroupStatistics, self).__init__(data)
        self.groups = groups
        self.results = []
        if not operations.is_iterable(data):
            pass
        else:
            if operations.is_dict(data):
                self.groups = data.keys()
                self.data = data.values()
            elif groups is None:
                self.groups = range(1, len(data) + 1)
            self.logic()

    def logic(self):
        for i, d in enumerate(self.data):
            if len(d) == 0:
                self.groups = self.groups[:i] + self.groups[i + 1:]
                continue
            else:
                if not operations.is_vector(d):
                    d = vector.Vector(d)
                if len(d) < self.__min_size:
                    self.groups = self.groups[:i] + self.groups[i + 1:]
                    continue
                self.results.append(self.run(d, self.groups[i]))
        if len(self.results) > 0:
            self.output()

    def run(self, vector, group):
        count = len(vector)
        avg = mean(vector)
        sd = std(vector, ddof=1)
        vmax = amax(vector)
        vmin = amin(vector)
        q2 = median(vector)
        return {"group": group, "count": count, "mean": avg, "std": sd, "max": vmax, "median": q2, "min": vmin}

    def output(self):
        size = 12
        header = ""
        line = ""
        offset = 0
        shift = False
        spacing = "{:.5f}"
        labels = ["Count", "Mean", "Std.", "Min", "Q2", "Max", "Group"]

        for s in labels:
            header = header + s + " " * (size - len(s))
        print header
        print "-" * len(header)
        for v in self.results:
            stats = [str(v["count"]),
                     spacing.format(v["mean"]),
                     spacing.format(v["std"]),
                     spacing.format(v["min"]),
                     spacing.format(v["median"]),
                     spacing.format(v["max"]),
                     str(v["group"])
                     ]
            for i, s in enumerate(stats):
                if offset == 1 or shift:
                    offset = -1
                    shift = False
                else:
                    offset = 0
                try:
                    if stats[i + 1][0] == "-":
                        if offset == -1:
                            offset = 0
                            shift = True
                        else:
                            offset = 1
                    line = line + s + " " * (size - offset - len(s))
                except IndexError:
                    line = line + s + " " * (size - offset - len(s))
            print line
            line = ""
