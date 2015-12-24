"""Module: analysis.py
Classes:
    Analysis - Generic analysis root class.
    Test - Generic statistical test class.
    GroupTest - Perform a test on multiple vectors that are passed as a tuple of arbitrary length.
    Comparison - Perform a test on two independent vectors of equal length.
    NormTest - Tests for whether data is normally distributed or not.
    GroupNormTest - Tests a group of data to see if they are normally distributed or not.
    TTest - Performs a T-Test on the two provided vectors.
    LinearRegression - Performs a linear regression between two vectors.
    Correlation - Performs a pearson or spearman correlation between two vectors.
    Anova - Performs a one-way ANOVA on a group of vectors.
    Kruskal - Performs a non-parametric Kruskal-Wallis test on a group of vectors.
    EqualVariance - Checks a group of vectors for equal variance.
    VectorStatistics - Reports basic summary stats for a provided vector.
    GroupStatistics - Reports basic summary stats for a group of vectors.
Functions:
    analyze - Magic method for performing quick data analysis.
"""
# Python3 compatability
from __future__ import absolute_import
from __future__ import print_function

# Scipy imports
from scipy.stats import linregress, shapiro, pearsonr, spearmanr, ttest_ind, \
    ttest_1samp, f_oneway, kruskal, bartlett, levene, skew, kurtosis

# Numpy imports
from numpy import concatenate, mean, std, median, amin, amax, percentile

# Local imports
from ..data.vector import Vector
from ..operations.data_operations import is_dict, is_iterable, is_vector, is_group,\
    is_dict_group, drop_nan, drop_nan_intersect
from ..graphs.graph import GraphHisto, GraphScatter, GraphBoxplot


class Analysis(object):
    """Generic analysis root class.

    Members:
        data - the data used for analysis.
        display - flag for whether to display the analysis output.
        results - a tuple representing the results of the analysis.

    Methods:
        logic - This method needs to run the analysis, set the results member, and display the output at bare minimum.
        run - This method should return the results of the specific analysis.
        output - This method shouldn't return a value and only produce a side-effect.
    """

    def __init__(self, data, display=True):
        """Initialize the data and results members.

        Override this method to initialize additional members or perform
        checks on data.
        """
        self.data = data
        self.display = display
        self.results = 0.0

    def logic(self):
        """This method needs to run the analysis, set the results member, and
        display the output at bare minimum.

        Override this method to modify the execution sequence of the analysis.
        """
        self.results = self.run()
        if self.display:
            self.output()

    def run(self):
        """This method should return the results of the specific analysis.

        Override this method to perform a specific analysis or calculation.
        """
        return 0.0

    def output(self):
        """This method shouldn't return a value and only produce a side-effect.

        Override this method to write the formatted output to std out.
        """
        print(self.results)
        pass

    def __str__(self):
        return str(self.results)

    def __repr__(self):
        return str(self.results)


class Test(Analysis):
    """Generic statistical test class.
    Members:
        data - the data used for analysis.
        display - flag for whether to display the analysis output.
        results - a tuple representing the results of the analysis.
        alpha - the statistical significance of the test.
    Methods:
        logic - If the result is greater than the significance, print the null hypothesis, otherwise,
            the alternate hypothesis.
        run - This method should return the results of the specific analysis.
        output - This method shouldn't return a value and only produce a side-effect.
        h0 - Prints the null hypothesis.
        ha - Prints the alternate hypothesis.
    """

    def __init__(self, data, alpha=0.05, display=True):

        super(Test, self).__init__(data, display=display)

        # Set members
        self.alpha = alpha
        self.results = 1, 0

        # If data is not a vector, wrap it in a Vector object
        if not is_vector(data):
            self.data = Vector(data)

        # Stop the test if the vector is empty
        if self.data.is_empty():
            print("vector is empty")
            pass
        else:

            # Remove NaN values from the vector
            self.data = drop_nan(self.data)

            # Run the test and display the results
            self.logic()

    def logic(self):
        self.results = self.run()
        if self.display:
            self.output()

            # If the result is greater than the significance, print the null
            # hypothesis, otherwise, the alternate hypothesis
            if self.results[0] > self.alpha:
                self.h0()
            else:
                self.ha()
            print("")

    def run(self):
        """ The default p-value is 1
        """
        return 1, 0

    def output(self):
        print(str(self.results[1]) + ", " + str(self.results[0]))

    def h0(self):
        print("H0: ")

    def ha(self):
        print("HA: ")


class GroupTest(Test):
    """ Perform a test on multiple vectors that are passed as a tuple of arbitrary length.
    """

    def __init__(self, *groups, **parms):
        self.alpha = 0.05
        self.data = []
        self.display = True
        self.results = 1, 0

        self.__dict__.update(parms)

        if is_dict(groups[0]):
            groups = list(groups[0].values())
        for group in groups:
            if not is_vector(group):
                group = drop_nan(Vector(group))
            if group.is_empty():
                continue
            if len(group) == 1:
                continue
            self.data.append(group.data)
        super(GroupTest, self).logic()


class Comparison(Test):
    """Perform a test on two independent vectors of equal length."""

    __min_size = 2

    def __init__(self, xdata, ydata, alpha=0.05, display=True):

        self.xdata = xdata
        self.ydata = ydata
        self.alpha = alpha
        self.display = display
        self.results = 1, 0

        if not is_vector(xdata):
            self.xdata = Vector(xdata)
        if not is_vector(ydata):
            self.ydata = Vector(ydata)
        if len(xdata) != len(ydata):
            print("Vector lengths are not equal")
            pass
        elif self.xdata.is_empty() or self.ydata.is_empty():
            print("At least one vector is empty")
            pass
        else:
            self.xdata, self.ydata = drop_nan_intersect(self.xdata, self.ydata)
        self.logic()

    def logic(self):
        if len(self.xdata) <= self.__min_size or len(self.ydata) <= self.__min_size:
            return self.results
        super(Comparison, self).logic()


class NormTest(Test):
    """Tests for whether data is normally distributed or not."""

    def run(self):
        w_value, p_value = shapiro(self.data)
        return p_value, w_value

    def output(self):
        name = "Shapiro-Wilk test for normality"
        print("")
        print(name)
        print("-" * len(name))
        print("")
        print("W value = " + "{:.4f}".format(self.results[1]))
        print("p value = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: Data is normally distributed")

    def ha(self):
        print("HA: Data is not normally distributed")


class GroupNormTest(GroupTest):
    """Tests a group of data to see if they are normally distributed or not."""

    def run(self):
        w_value, p_value = shapiro(concatenate(self.data))
        return p_value, w_value

    def output(self):
        name = "Shapiro-Wilk test for normality"
        print("")
        print(name)
        print("-" * len(name))
        print("")
        print("W value = " + "{:.4f}".format(self.results[1]))
        print("p value = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: Data is normally distributed")

    def ha(self):
        print("HA: Data is not normally distributed")


class TTest(Test):
    """Performs a T-Test on the two provided vectors."""

    def __init__(self, xdata, ydata, alpha=0.05, display=True):

        self.alpha = alpha
        self.display = display

        if is_iterable(ydata):
            # If data is not a vector, wrap it in a Vector object
            if not is_vector(ydata):
                self.ydata = Vector(ydata)

            # Stop the test if the vector is empty
            if self.ydata.is_empty():
                print("vector is empty")
                pass
            else:
                # Remove NaN values from the vector
                self.ydata = drop_nan(self.ydata)
        else:
            try:
                self.ydata = float(ydata)
            except (ValueError, TypeError):
                print("ydata is not a vector or a number")
                pass

        super(TTest, self).__init__(xdata, alpha=alpha, display=display)

    def run(self):

        if is_iterable(self.ydata):
            if EqualVariance(self.data, self.ydata, display=False).results[0] > self.alpha:
                t, p = ttest_ind(self.data, self.ydata, equal_var=True)
                test = "T Test"
            else:
                t, p = ttest_ind(self.data, self.ydata, equal_var=False)
                test = "Welch's T Test"
        else:
            t, p = ttest_1samp(self.data, float(self.ydata))
            test = "1 Sample T Test"
        return float(p), float(t), test

    def output(self):
        print("")
        print(self.results[2])
        print("-" * len(self.results[2]))
        print("")
        print("t = " + "{:.4f}".format(self.results[1]))
        print("p = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: Means are matched")

    def ha(self):
        print("HA: Means are significantly different")


class LinearRegression(Comparison):
    """Performs a linear regression between two vectors."""

    __min_size = 3

    def run(self):
        slope, intercept, r2, p_value, std_err = linregress(self.xdata, self.ydata)
        count = len(self.xdata)
        return p_value, slope, intercept, r2, std_err, count

    def output(self):
        name = "Linear Regression"
        print("")
        print(name)
        print("-" * len(name))
        print("")
        print("count     = " + str(self.results[5]))
        print("slope     = " + "{:.4f}".format(self.results[1]))
        print("intercept = " + "{:.4f}".format(self.results[2]))
        print("R^2       = " + "{:.4f}".format(self.results[3]))
        print("std err   = " + "{:.4f}".format(self.results[4]))
        print("p value   = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: There is no significant relationship between predictor and response")

    def ha(self):
        print("HA: There is a significant relationship between predictor and response")


class Correlation(Comparison):
    """Performs a pearson or spearman correlation between two vectors."""

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
        print("")
        print(name)
        print("-" * len(name))
        print("")
        if self.results[2] == "pearson":
            print("Pearson Coeff:")
        else:
            print("Spearman Coeff:")
        print("r = " + "{:.4f}".format(self.results[1]))
        print("p = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: There is no significant relationship between predictor and response")

    def ha(self):
        print("HA: There is a significant relationship between predictor and response")


class Anova(GroupTest):
    """Performs a one-way ANOVA on a group of vectors."""

    __min_size = 2

    def run(self):
        if len(self.data) <= self.__min_size:
            return self.results
        f_value, p_value = f_oneway(*tuple(self.data))
        return p_value, f_value

    def output(self):
        name = "Oneway ANOVA"
        print("")
        print(name)
        print("-" * len(name))
        print("")
        print("f value = " + "{:.4f}".format(self.results[1]))
        print("p value = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: Group means are matched")

    def ha(self):
        print("HA: Group means are not matched")


class Kruskal(GroupTest):
    """Performs a non-parametric Kruskal-Wallis test on a group of vectors."""

    __min_size = 2

    def run(self):
        if len(self.data) <= self.__min_size:
            return self.results
        h_value, p_value = kruskal(*tuple(self.data))
        return p_value, h_value

    def output(self):
        name = "Kruskal-Wallis"
        print("")
        print(name)
        print("-" * len(name))
        print("")
        print("H value = " + "{:.4f}".format(self.results[1]))
        print("p value = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: Group means are matched")

    def ha(self):
        print("HA: Group means are not matched")


class EqualVariance(GroupTest):
    """Checks a group of vectors for equal variance."""

    __min_size = 2

    def run(self):
        if len(self.data) < self.__min_size:
            return self.results
        if NormTest(concatenate(self.data), display=False, alpha=self.alpha).results[0] > self.alpha:
            statistic, p_value = bartlett(*tuple(self.data))
            t = "Bartlett Test"
        else:
            statistic, p_value = levene(*tuple(self.data))
            t = "Levene Test"
        return p_value, statistic, t

    def output(self):
        print("")
        print(self.results[2])
        print("-" * len(self.results[2]))
        print("")
        if self.results[2] == "Bartlett Test":
            print("T value = " + "{:.4f}".format(self.results[1]))
        else:
            print("W value = " + "{:.4f}".format(self.results[1]))
        print("p value = " + "{:.4f}".format(self.results[0]))
        print("")

    def h0(self):
        print("H0: Variances are equal")

    def ha(self):
        print("HA: Variances are not equal")


class VectorStatistics(Analysis):
    """Reports basic summary stats for a provided vector."""

    __min_size = 2

    def __init__(self, data, sample=False, display=True):
        super(VectorStatistics, self).__init__(data, display=display)

        self.results = None
        self.sample = sample

        self.data = drop_nan(Vector(data))

        if self.data.is_empty():
            print("vector is empty")
            pass
        elif len(self.data) < self.__min_size:
            pass
        else:
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
        print("")
        print(name)
        print("-" * len(name))
        print("")
        print("Count    = " + str(self.results["count"]))
        print("Mean     = " + "{:.3f}".format(self.results['mean']))
        print("Std Dev  = " + "{:.3f}".format(self.results['std']))
        print("Skewness = " + "{:.3f}".format(self.results['skew']))
        print("Kurtosis = " + "{:.3f}".format(self.results['kurtosis']))
        print("Max      = " + "{:.3f}".format(self.results['max']))
        print("75%      = " + "{:.3f}".format(self.results['q3']))
        print("50%      = " + "{:.3f}".format(self.results['median']))
        print("25%      = " + "{:.3f}".format(self.results['q1']))
        print("Min      = " + "{:.3f}".format(self.results['min']))
        print("IQR      = " + "{:.3f}".format(self.results['iqr']))
        print("Range    = " + "{:.3f}".format(self.results['range']))
        print("")


class GroupStatistics(Analysis):
    """Reports basic summary stats for a group of vectors."""

    __min_size = 1

    def __init__(self, data, groups=None, display=True):
        super(GroupStatistics, self).__init__(data, display=display)
        self.groups = groups
        self.results = []
        if not is_iterable(data):
            pass
        else:
            if is_dict(data):
                self.groups = list(data.keys())
                self.data = list(data.values())
            elif groups is None:
                self.groups = list(range(1, len(data) + 1))
            self.logic()

    def logic(self):
        for i, d in enumerate(self.data):
            if len(d) == 0:
                self.groups = self.groups[:i] + self.groups[i + 1:]
                continue
            else:
                if not is_vector(d):
                    d = Vector(d)
                if len(d) < self.__min_size:
                    self.groups = self.groups[:i] + self.groups[i + 1:]
                    continue
                self.results.append(self.run(d, self.groups[i]))
        if len(self.results) > 0:
            self.output()

    def run(self, vector, group):
        vector = drop_nan(vector)
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
        print(header)
        print("-" * len(header))
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
            print(line)
            line = ""


def analyze(
        xdata,
        ydata=None,
        groups=None,
        name=None,
        xname=None,
        yname=None,
        alpha=0.05,
        categories='Categories'):
    """Magic method for performing quick data analysis.

    :param xdata: A Vector, numPy Array or sequence like object
    :param ydata: An optional secondary Vector, numPy Array or sequence object
    :param groups: A list of group names. The box plots will be graphed in order of groups
    :param name: The response variable label
    :param xname: The predictor variable (x-axis) label
    :param yname: The response variable (y-axis) label
    :param alpha: The significance level of the test
    :param categories: The x-axis label when performing a group analysis
    :return: A tuple of xdata and ydata
    """

    # Compare Group Means and Variance
    if is_group(xdata) or is_dict_group(xdata):
        if is_dict(xdata):
            groups = list(xdata.keys())
            xdata = list(xdata.values())

        # Apply the y data label
        if yname:
            yname = yname
        elif name:
            yname = name
        else:
            yname = 'Values'

        # Apply the x data label
        if xname:
            label = xname
        else:
            label = categories

        # Show the box plot and stats
        GraphBoxplot(xdata, groups, label, yname=yname)
        GroupStatistics(xdata, groups)
        p = EqualVariance(*xdata).results[0]

        # If normally distributed and variances are equal, perform one-way ANOVA
        # Otherwise, perform a non-parametric Kruskal-Wallis test
        if GroupNormTest(*xdata, display=False, alpha=alpha).results[0] > alpha and p > alpha:
            if len(xdata) == 2:
                TTest(xdata[0], xdata[1])
            else:
                Anova(*xdata)
        else:
            if len(xdata) == 2:
                TTest(xdata[0], xdata[1])
            else:
                Kruskal(*xdata)
        pass

    # Correlation and Linear Regression
    elif is_iterable(xdata) and is_iterable(ydata):

        # Apply the x data label
        label = 'Predictor'
        if xname:
            label = xname

        # Apply the y data label
        if yname:
            yname = yname
        elif name:
            yname = name
        else:
            yname = 'Response'

        # Convert xdata and ydata to Vectors
        if not is_vector(xdata):
            xdata = Vector(xdata)
        if not is_vector(ydata):
            ydata = Vector(ydata)

        # Show the scatter plot, correlation and regression stats
        GraphScatter(xdata, ydata, label, yname)
        LinearRegression(xdata, ydata)
        Correlation(xdata, ydata)
        pass

    # Histogram and Basic Stats
    elif is_iterable(xdata):

        # Apply the data label
        label = 'Data'
        if name:
            label = name
        elif xname:
            label = xname

        # Convert xdata to a Vector
        if not is_vector(xdata):
            xdata = Vector(xdata)

        # Show the histogram and stats
        GraphHisto(xdata, name=label)
        VectorStatistics(xdata)
        NormTest(xdata, alpha=alpha)
        pass
    else:
        return xdata, ydata
