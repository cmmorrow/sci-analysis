# Data analysis module

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# Perform a one_way anova on all arguments(groups).
# Assumption is arguments have equal variance, are normally distributed and > 2
# ------------------------------------------------------------------------------
def anova(*groups, **parms):
    alpha = 0.05
    copy = []

    parm_list = sorted(parms.keys())
    for parm in parm_list:
        if parm == "alpha":
            alpha = parm

    if isdict(groups[0]):
        groups = groups[0].values()
    for group in groups:
        if len(group) < 2:
            continue
        v = clean(group)
        if len(v) < 2:
            continue
        copy.append(v)
    if len(copy) < 3:
        return 0, 0
    f_value, p_value = st.f_oneway(*tuple(copy))
    print "ANOVA"
    print "-" * 8
    print "f value = " + "{:.4f}".format(f_value)
    print "p value = " + "{:.4f}".format(p_value)
    if p_value < alpha:
        print "HA: Group means are not matched"
    else:
        print "H0: Group means are matched"
    print ""
    return f_value, p_value


# Perform a non-parametric one_way on all arguments(groups).
# Assumption is arguments > 2
# ------------------------------------------------------------
def kruskal(*groups, **parms):
    alpha = 0.05
    copy = []

    parm_list = sorted(parms.keys())
    for parm in parm_list:
        if parm == 'alpha':
            alpha = parm

    if isdict(groups[0]):
        groups = groups[0].values()
    for group in groups:
        if len(group) < 2:
            continue
        v = clean(group)
        if len(v) < 2:
            continue
        copy.append(v)
    if len(copy) < 3:
        return 0, 0
    h_value, p_value = st.kruskal(*tuple(copy))
    print "Kruskal-Wallis"
    print "-" * 8
    print "H value = " + "{:.4f}".format(h_value)
    print "p value = " + "{:.4f}".format(p_value)
    if p_value < alpha:
        print "HA: Group means are not matched"
    else:
        print "H0: Group means are matched"
    print ""
    return h_value, p_value


# Tests if data is normally distributed
# ----------------------------------------
def norm_test(data, alpha=0.05, display=True):
    if any(is_iterable(i) for i in data):
        data = np.concatenate(data)
    x = clean(data)
    if x.size < 4:
        return 0, 0
    w_value, p_value = st.shapiro(x)
    if display:
        print "Shapiro-Wilk test for normality"
        print "-" * 8
        print "W value = " + "{:.4f}".format(w_value)
        print "p value = " + "{:.4f}".format(p_value)
        if p_value < alpha:
            print "HA: Data is not normally distributed"
        else:
            print "H0: Data is normally distributed"
        print ""
    return w_value, p_value


# Tests to see if arguments(groups) have equal variance
# ------------------------------------------------------
def equal_variance(*groups, **parms):
    alpha = 0.05
    copy = []

    parm_list = sorted(parms.keys())
    for parm in parm_list:
        if parm == 'alpha':
            alpha = parm

    if isdict(groups[0]):
        groups = groups[0].values()
    for group in groups:
        if len(group) < 2:
            continue
        v = clean(group)
        if len(v) < 2:
            continue
        copy.append(v)
    if len(copy) < 3:
        return 0, 0
    if norm_test(copy, display=False)[1] < alpha:
        # print "Data is not normally distributed"
        statistic, p_value = st.levene(*tuple(copy))
        print "Levene Test"
        print "-" * 8
        print "W value = " + "{:.4f}".format(statistic)
        print "p value = " + "{:.4f}".format(p_value)
        if p_value < alpha:
            print "HA: Variances are not equal"
        else:
            print "H0: Variances are equal"
        print ""
    else:
        # print "Data is normal"
        statistic, p_value = st.bartlett(*tuple(copy))
        print "Bartlett Test"
        print "-" * 8
        print "T value = " + "{:.4f}".format(statistic)
        print "p value = " + "{:.4f}".format(p_value)
        if p_value < alpha:
            print "HA: Variances are not equal"
        else:
            print "H0: Variances are equal"
        print ""
    return statistic, p_value


# Performs Student's T-test
# If variances are not equal, performs Welch's T-test
# If ydata is not a collection, a 1 sample T-test is performed
# --------------------------------------------------------------
def t_test(xdata, ydata, alpha=0.05):
    x = clean(xdata)
    if is_iterable(ydata):
        y = clean(ydata)
        if equal_variance(xdata, ydata)[1] < alpha:
            t, p = st.ttest_ind(x, y, equal_var=True)
            print "t-test"
        else:
            t, p = st.ttest_ind(x, y, equal_var=False)
            print "Welch's t-test"
    else:
        t, p = st.ttest_1samp(x, float(ydata))
        print "1 Sample t-test"
    print "-" * 8
    print "t = " + str(t)
    print "p = " + str(p)
    if p < alpha:
        print "Reject H0: Means are different"
    else:
        print "H0: Means are matched"
    print ""
    return t, p


# Performs a linear regression on ydata as dependent on xdata
# -------------------------------------------------------------
def linear_regression(xdata, ydata, alpha=0.05):
    x, y = clean(xdata, ydata)
    if x.size < 4 or y.size < 4:
        return 0, 0, 0, 0, 0
    slope, intercept, r2, p_value, std_err = st.linregress(x, y)
    print "Linear Regression"
    print "-" * 8
    print "slope = " + "{:.4f}".format(slope)
    print "intercept = " + "{:.4f}".format(intercept)
    print "R^2 = " + "{:.4f}".format(r2)
    print "p = " + "{:.4f}".format(p_value)
    print "std err = " + "{:.4f}".format(std_err)
    if p_value < alpha:
        print "HA: The relationship between predictor and response is significant"
    else:
        print "H0: There is no significant relationship between predictor and response"
    print ""
    return slope, intercept, r2, p_value, std_err


# Calculates the Pearson correlation coefficient between xdata and ydata
# If xdata and ydata are not normally distributed, Spearman's is used
# ------------------------------------------------------------------------
def correlate(xdata, ydata, alpha=0.05):
    x, y = clean(xdata, ydata)
    if x.size < 4 or y.size < 4:
        return 0, 0
    print "Correlation"
    print "-" * 8
#    if norm_test(x, display=False, alpha=alpha)[1] > alpha and norm_test(y, display=False, alpha=alpha)[1] > alpha:
    if norm_test(np.concatenate([x, y]), display=False, alpha=alpha)[1] > alpha:
        r, p = st.pearsonr(x, y)
        print "Pearson Coeff:"
    else:
        r, p = st.spearmanr(x, y)
        print "Spearman Coeff:"
    print "r = " + "{:.4f}".format(r)
    print "p = " + "{:.4f}".format(p)
    if p < alpha:
        print "HA: There is a relationship between response and predictor"
    else:
        print "H0: There is no significant relationship between response and predictor"
    print ""
    return r, p


# Calculates and displays basic stats of data
# ---------------------------------------------
def statistics(data, sample=True):
    v = clean(data)
    dof = 0
    if v.size > 1:
        if sample:
            dof = 1
#        count, (vmin, vmax), mean, variance, skew, kurt = st.describe(v, ddof=dof)
        count = v.size
        mean = np.mean(v)
        std = np.std(v, ddof=dof)
        #std = np.sqrt(variance)
        median = np.median(v)
        vmin = np.amin(v)
        vmax = np.amax(v)
        vrange = vmax - vmin
        skew = st.skew(v)
        kurt = st.kurtosis(v)
        q1 = np.percentile(v, 25)
        q3 = np.percentile(v, 75)
        iqr = q3 - q1
        print "Statistics"
        print "-" * 8
        print "Count = " + str(count)
        print "Mean = " + str(mean)
        print "Standard Deviation = " + str(std)
        print "Skewness = " + str(skew)
        print "Kurtosis = " + str(kurt)
        print "Max = " + str(vmax)
        print "75% = " + str(q3)
        print "50% = " + str(median)
        print "25% = " + str(q1)
        print "Min = " + str(vmin)
        print "IQR = " + str(iqr)
        print "Range = " + str(vrange)
        print ""
        return count, mean, std, skew, kurt, vmax, q3, median, q1, vmin, iqr, vrange
    return 1, v[0], 0, 0, 0, v[0], v[0], v[0], v[0], v[0], 0, 0


# Removes NaN data from numPy arrays xdata and ydata, only preserving values where xdata and ydata are True
# ---------------------------------------------------------------------------------------------------------
def dropnan_intersect(xdata, ydata):
    c = np.logical_and(~np.isnan(xdata), ~np.isnan(ydata))
    return xdata[c], ydata[c]


# Removes NaN values from numPy array data
# ----------------------------------------
def dropnan(data):
    return data[~np.isnan(data)]


# Removes strings in data and returns data as an array of floats
# --------------------------------------------------------------
def strip(data):
    if not isarray(data):
        for i in range(len(data)):
            try:
                data[i] = float(data[i])
            except ValueError:
                data[i] = float("nan")
            except KeyError:
                data = strip(data.values())
        data = np.array(data)
    data.astype('float')
    return data


# Tests if data is a collection and not empty
# --------------------------------------------
def is_iterable(data):
    try:
        if len(data) > 0:
            return True
        else:
            return False
    except TypeError:
        return False


# Tests if data is a numPy array object
# -------------------------------------
def isarray(data):
    try:
        data.shape
        return True
    except AttributeError:
        return False

# Test if data is a dictionary object
# -----------------------------------
def isdict(data):
    try:
        data.keys()
        return True
    except AttributeError:
        return False

# Cleans the data and returns a numPy array-like object
# -------------------------------------------------------
def clean(xdata, ydata = []):
    # TODO: Check xdata and ydata object type for equivalence
    if len(ydata) > 0:
        return dropnan_intersect(strip(xdata), strip(ydata))
    else:
        return dropnan(strip(xdata))


# Returns the corresponding color tuple based on a numeric index
# ----------------------------------------------------------------
def get_color(num):
    colors = [(0, 0, 1, 1),
              (0, 0.5, 0, 1),
              (1, 0, 0, 1),
              (0, 1, 1, 1),
              (1, 1, 0, 1),
              (1, 0, 1, 1),
              (1, 0.5, 0, 1),
              (0.5, 0, 1, 1),
              (0.5, 1, 0, 1),
              (1, 1, 1, 1)
              ]
    desired_color = []
    if num < 0:
        num *= -1
    floor = int(num) / len(colors)
    remainder = int(num) % len(colors)
    selected = colors[remainder]
    if floor > 0:
        for value in selected:
            desired_color.append(value / (2.0 * floor) + 0.4)
        return tuple(desired_color)
    else:
        return selected


# Slices groups at the given index
# -----------------------------------
def slice_group(groups, index):
    return groups[:index] + groups[index + 1:]


# Calculates basic stats for each specified group in groups
# ----------------------------------------------------------
def group_stats(data, groups):
    if not is_iterable(data):
        pass
    else:
        if isdict(data):
            groups = data.keys()
            data = data.values()
        if not groups:
            groups = range(1, len(data) + 1)
        print "Count     Mean      Std.      Max       50%       Min       Group"
        print "----------------------------------------------------------------------"
        for i, d in enumerate(data):
            if len(d) == 0:
                groups = slice_group(groups, i)
                continue
            else:
                cleaned = clean(d)
                if len(cleaned) == 0:
                    groups = slice_group(groups, i)
                    continue
                count = len(cleaned)
                mean = "{:.3f}".format(np.mean(cleaned))
                std = "{:.3f}".format(np.std(cleaned, ddof=1))
                vmax = "{:.3f}".format(np.amax(cleaned))
                median = "{:.3f}".format(np.median(cleaned))
                vmin = "{:.3f}".format(np.amin(cleaned))
                print "{:<10}".format(count) + "{:<10}".format(mean) + "{:<10}".format(std) + "{:<10}".format(
                    vmax) + "{:<10}".format(median) + "{:<10}".format(vmin) + "{:<10}".format(groups[i])
        print ""
        pass


# Displays a histogram of data
# ------------------------------
def graph_histo(data, bins=20, name='Data', color='green', boxplot=True):
    x = clean(data)
    plt.figure(figsize=(5, 5))
    if len(x) < bins:
        bins = len(x)
    if boxplot:
        plt.subplot(211)
        plt.grid(plt.boxplot(x, vert=False, showmeans=True), which='major')
        plt.subplot(212)
    if bins > len(x):
        bins = len(x)
    plt.grid(plt.hist(x, bins, normed=True, color=color))
    plt.ylabel('Probability')
    plt.xlabel(name)
    plt.show()
    pass


# Displays a scatter plot of xdata and ydata
# ---------------------------------------------
def graph_scatter(xdata, ydata, xname='x', yname='y', fit=True, pointstyle='k.', linestyle='r-'):
    x, y = clean(xdata, ydata)
    p = np.polyfit(x, y, 1, full=True)
    plt.grid(plt.plot(x, y, pointstyle))
    if fit:
        plt.plot(x, np.polyval(p[0], x), linestyle)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()
    pass


# Displays box plots and a probability plot of values in each group in groups
# ----------------------------------------------------------------------------
def graph_boxplot(values, groups=[], xname='Values', categories='Categories', probplot=True):
    if not is_iterable(values):
        pass
    else:
        v = []
        prob = []
        if isdict(values):
            groups = values.keys()
            values = values.values()
        if not groups:

            # Create numberic group names if not specified
            groups = range(1, len(values) + 1)
        for i, value in enumerate(values):

            # If a group is null, don't display it
            if len(value) == 0:
                groups = slice_group(groups, i)
                continue
            else:
                cleaned = clean(value)
                if len(cleaned) == 0:
                    groups = slice_group(groups, i)
                    continue
                v.append(cleaned)
                if probplot:
                    q, fit = st.probplot(cleaned)
                    prob.append((q, fit))
        if probplot:
            plt.figure(figsize=(15, 5))
            plt.subplot(122)
            plt.grid()
            for i, g in enumerate(prob):
                plt.plot(g[0][0], g[0][1], marker='^', color=get_color(i), label=groups[i])
                plt.plot(g[0][0], g[1][0] * g[0][0] + g[1][1], linestyle='-', color=get_color(i))
            plt.legend(loc='best')
            plt.xlabel("Quantiles")
            plt.subplot(121)
        else:
            plt.figure()
        plt.grid(plt.boxplot(v, showmeans=True, labels=groups), which='major')
        plt.ylabel(xname)
        plt.xlabel(categories)
        plt.show()
        pass


# Main function that will perform an analysis based on the provided data
# ------------------------------------------------------------------------
def analyze(xdata, ydata=[], groups=[], name='', xname='', yname='y', alpha=0.05, categories='Categories'):

    # Compare Group Means and Variance
    if any(is_iterable(x) for x in xdata):

        if isdict(xdata):
            groups = xdata.keys()
            xdata = xdata.values()

        # Apply the x data label
        label = 'x'
        if xname:
            label = xname

        # Show the box plot and stats
        graph_boxplot(xdata, groups, label, categories)
        group_stats(xdata, groups)
        stat, p = equal_variance(*xdata)

        # If normally distributed and variances are equal, perform one-way ANOVA
        # Otherwise, perform a non-parametric Kruskal-Wallis test
        if norm_test(xdata, display=False)[1] > alpha and p > alpha:
            anova(*xdata)
        else:
            kruskal(*xdata)
        pass

    # Correlation and Linear Regression
    elif is_iterable(xdata) and is_iterable(ydata):

        # Apply the x data label
        label = 'x'
        if xname:
            label = xname

        # Show the scatter plot, correlation and regression stats
        graph_scatter(xdata, ydata, label, yname)
        correlate(xdata, ydata)
        linear_regression(xdata, ydata)
        pass

    # Histogram and Basic Stats
    elif is_iterable(xdata):

        # Apply the data label
        label = 'Data'
        if name:
            label = name
        elif xname:
            label = xname

        # Show the histogram and stats
        graph_histo(xdata, name=label)
        statistics(xdata)
        norm_test(xdata)
        pass
    else:
        return xdata, ydata