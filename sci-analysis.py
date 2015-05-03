# Data analysis module

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def anova(*groups):
    alpha = 0.05
    copy = []
    for group in groups:
        if len(group) == 0:
            continue
        v = dropnan(group)
        if len(v) == 0:
            continue
        copy.append(v)
    if len(copy) < 2:
        return 0, 0
    f_value, p_value = st.f_oneway(*tuple(copy))
    print "ANOVA"
    print "--------"
    print "p value = " + "{:.4f}".format(p_value)
    print "f value = " + "{:.4f}".format(f_value)
    if p_value < alpha:
        print "Groups are not matched"
    else:
        print "Groups are matched"
    print ""
    return f_value, p_value


def kruskal(*groups):
    alpha = 0.05
    copy = []
    for group in groups:
        if len(group) == 0:
            continue
        v = dropnan(group)
        if len(v) == 0:
            continue
        copy.append(v)
    if len(copy) < 2:
        return 0, 0
    h_value, p_value = st.kruskal(*tuple(copy))
    print "Wilcoxon/Kruskal-Wallis"
    print "--------"
    print "p value = " + "{:.4f}".format(p_value)
    print "H value = " + "{:.4f}".format(h_value)
    if p_value < alpha:
        print "Groups are not matched"
    else:
        print "Groups are matched"
    print ""
    return h_value, p_value


def norm_test(data, alpha=0.05, display=True):
    if any(is_iterable(i) for i in data):
        data = np.concatenate(data)
    x = dropnan(data)
    if x.size < 3:
        return 0, 0
    w_value, p_value = st.shapiro(x)
    if display:
        print "Shapiro-Wilk test for normality"
        print "--------"
        print "p value = " + "{:.4f}".format(p_value)
        print "W value = " + "{:.4f}".format(w_value)
        if p_value < alpha:
            print "Data is not normally distributed"
        else:
            print "Data is normally distributed"
        print ""
    return w_value, p_value


def equal_variance(*groups):
    alpha = 0.05
    copy = []
    for group in groups:
        if len(group) == 0:
            continue
        v = dropnan(group)
        if len(v) == 0:
            continue
        copy.append(v)
    if len(copy) < 2:
        return 0, 0
    if norm_test(copy, display=False)[1] < alpha:
        # print "Data is not normally distributed"
        statistic, p_value = st.levene(*tuple(copy))
        print "Levene Test"
        print "-" * 8
        print "p value = " + "{:.4f}".format(p_value)
        print "W value = " + "{:.4f}".format(statistic)
        if p_value < alpha:
            print "Variances are not equal"
        else:
            print "Variances are equal"
        print ""
    else:
        # print "Data is normal"
        statistic, p_value = st.bartlett(*tuple(copy))
        print "Bartlett Test"
        print "-" * 8
        print "p value = " + "{:.4f}".format(p_value)
        print "T value = " + "{:.4f}".format(statistic)
        if p_value < alpha:
            print "Variances are not equal"
        else:
            print "Variances are equal"
        print ""
    return statistic, p_value


def t_test(xdata, ydata, alpha=0.05):
    x = dropnan(xdata)
    if is_iterable(ydata):
        y = dropnan(ydata)
        if equal_variance(xdata, ydata)[1] > alpha:
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
    print ""
    return t, p


def linear_regression(xdata, ydata):
    x, y = dropnan_intersect(xdata, ydata)
    if x.size < 3 or y.size < 3:
        return 0, 0, 0, 0, 0
    slope, intercept, r2, p_value, std_err = st.linregress(x, y)
    print "Linear Regression"
    print "-" * 8
    print "slope = " + "{:.4f}".format(slope)
    print "intercept = " + "{:.4f}".format(intercept)
    print "R^2 = " + "{:.4f}".format(r2)
    print "p = " + "{:.4f}".format(p_value)
    print "std err = " + "{:.4f}".format(std_err)
    print ""
    return slope, intercept, r2, p_value, std_err


def correlate(xdata, ydata, alpha=0.05):
    x, y = dropnan_intersect(xdata, ydata)
    if x.size < 3 or y.size < 3:
        return 0, 0
    print "Correlation"
    print "-" * 8
    if norm_test(x, display=False, alpha=alpha)[1] > alpha and norm_test(y, display=False, alpha=alpha)[1] > alpha:
        r, p = st.pearsonr(x, y)
        print "Pearson Coeff:"
    else:
        r, p = st.spearmanr(x, y)
        print "Spearman Coeff:"
    print "r = " + "{:.4f}".format(r)
    print "p = " + "{:.4f}".format(p)
    print ""
    return r, p


def statistics(data):
    v = dropnan(data)
    if v.size > 1:
        count, (vmin, vmax), mean, variance, skew, kurt = st.describe(v)
        # count = v.size
        #		mean = np.mean(v)
        #		std = np.std(v)
        std = np.sqrt(variance)
        median = np.median(v)
        #		vmin = np.amin(v)
        #		vmax = np.amax(v)
        vrange = vmax - vmin
        q1 = np.percentile(v, 25)
        q3 = np.percentile(v, 75)
        iqr = q3 - q1
        print "Statistics"
        print "--------"
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


def dropnan_intersect(xdata, ydata):
    x = np.asarray(strip(xdata))[~np.isnan(np.asarray(strip(ydata)))]
    x = x[~np.isnan(x)]
    y = np.asarray(strip(ydata))[~np.isnan(np.asarray(strip(xdata)))]
    y = y[~np.isnan(y)]
    return x, y


def dropnan(data):
    d = np.asarray(strip(data))
    return d[~np.isnan(d)]


def strip(data):
    try:
        data.shape
        clean = data
    except AttributeError:
        l = list(data)
        clean = filter(lambda x: type(x) == int or type(x) == float, l)
    return clean


def is_iterable(data):
    try:
        iter(data)
    except TypeError:
        return False
    else:
        if len(data) > 0:
            return True
        else:
            return False


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


def slice_group(groups, index):
    return groups[:index] + groups[index + 1:]


def group_stats(data, groups):
    if not is_iterable(data):
        pass
    else:
        if not groups:
            groups = range(1, len(data) + 1)
        print "Count     Mean      Std.      Max       50%       Min       Group"
        print "----------------------------------------------------------------------"
        for i, d in enumerate(data):
            if len(d) == 0:
                groups = slice_group(groups, i)
                continue
            else:
                clean = dropnan(d)
                if len(clean) == 0:
                    groups = slice_group(groups, i)
                    continue
                count = len(clean)
                mean = "{:.3f}".format(np.mean(clean))
                std = "{:.3f}".format(np.std(clean))
                vmax = "{:.3f}".format(np.amax(clean))
                median = "{:.3f}".format(np.median(clean))
                vmin = "{:.3f}".format(np.amin(clean))
                print "{:<10}".format(count) + "{:<10}".format(mean) + "{:<10}".format(std) + "{:<10}".format(
                    vmax) + "{:<10}".format(median) + "{:<10}".format(vmin) + "{:<10}".format(groups[i])
        print ""
        pass


def graph_histo(data, bins=20, name='Data', color='green', boxplot=True):
    x = dropnan(data)
    plt.figure(figsize=(5, 5))
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


def graph_scatter(xdata, ydata, xname='x', yname='y', fit=True, pointstyle='k.', linestyle='r-'):
    x, y = dropnan_intersect(xdata, ydata)
    p = np.polyfit(x, y, 1, full=True)
    plt.grid(plt.plot(x, y, pointstyle))
    if fit:
        plt.plot(x, np.polyval(p[0], x), linestyle)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.show()
    pass


def graph_boxplot(values, groups=[], xname='Values', categories='Categories', probplot=True):
    if not is_iterable(values):
        pass
    else:
        v = []
        prob = []
        # colors = ('blue', 'green', 'red', 'cyan', 'magenta', 'yellow')
        if not groups:
            groups = range(1, len(values) + 1)
        for i, value in enumerate(values):
            if len(value) == 0:
                groups = slice_group(groups, i)
                continue
            else:
                clean = dropnan(value)
                if len(clean) == 0:
                    groups = slice_group(groups, i)
                    continue
                v.append(clean)
                if probplot:
                    q, fit = st.probplot(clean)
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


def analyze(xdata, ydata=[], groups=[], xname='x', yname='y', alpha=0.05, categories='Categories'):
    if any(is_iterable(x) for x in xdata):
        graph_boxplot(xdata, groups, xname, categories)
        group_stats(xdata, groups)
        equal_variance(*xdata)
        if norm_test(xdata, display=False)[1] < alpha:
            kruskal(*xdata)
        else:
            anova(*xdata)
        pass
    elif is_iterable(xdata) and is_iterable(ydata):
        graph_scatter(xdata, ydata, xname, yname)
        correlate(xdata, ydata)
        linear_regression(xdata, ydata)
        pass
    elif is_iterable(xdata):
        graph_histo(xdata, name=xname)
        statistics(xdata)
        norm_test(xdata)
        pass
    else:
        return xdata, ydata