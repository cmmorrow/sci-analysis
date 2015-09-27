# sci-analysis
A light weight python data exploration and analysis tool by Chris Morrow

## Current Version:
1.3 --- Released September 21, 2015

### What is sci-analysis?
sci-analysis is a python package for performing rapid statistical data analysis. It provides a graphical representation of the supplied data as well as the statistical analysis. sci-analysis is smart enough to determine the correct analysis and tests to perform based on the shape of the data you provide, as well as whether the data is normally distributed.

Currently, sci-analysis can only be used for analyzing numeric data. Categorical data analysis is planned for a future version. The three types of analysis that can be performed are histograms of single vectors, correlation between two vectors and one-way ANOVA.

### Getting started with sci-analysis
Before using sci-analysis, be sure the following three packages are installed:
	* numpy
	* scipy
	* matplotlib
	
sci-analysis is also compatible with pandas and works best in the iPython Notebook.

First, download to your PC or clone the repo at: 
https://github.com/cmmorrow/sci-analysis

Next, add the sci-analysis directory to your project with:

```python
sys.path.extend(['<path to directory>/sci_analysis'])
import sci_analysis as a
import numpy as np
```

**Note:** The package name is `sci_analysis` with an underscore.

This will tell python where to find sci-analysis and import it to your project is the object `a`. 

If you are using the iPython Notebook, you will also want to use the following code instead to enable inline plots:

```python
%matplotlib inline
import matplotlib
sys.path.extend(['<path to directory>/sci_analysis'])
import sci_analysis as a
import numpy as np
```

Now, sci-analysis should be ready to use. Try the following code:

```python
a.analyze(np.random.randn(100))
```

A histogram and box plot of the data should appear, as well as printed output similar to that below:

```
Statistics
--------
Count = 100
Mean = -0.0346394170379
Standard Deviation = 1.00138009977
Skewness = 0.246797356486
Kurtosis = 0.0301715149203
Max = 2.98521191579
75% = 0.618195797909
50% = -0.1045351866
25% = -0.760766375821
Min = -2.43834596493
IQR = 1.37896217373
Range = 5.42355788072

Shapiro-Wilk test for normality
--------
W value = 0.9944
p value = 0.9581
H0: Data is normally distributed
```

You should probably note that numpy was only imported for the purpose of the above example. sci-analysis uses numpy internally, so it isn't necessary to import it unless you want to explicitly use it. 

### How do I use sci-analysis?

The easiest and fastest way to use sci-analysis is to call it's `analyze` function. Here's the signature for the `analyze` function:

```python
def analyze(
        xdata,
        ydata=None,
        groups=None,
        name=None,
        xname=None,
        yname=None,
        alpha=0.05,
        categories='Categories'):
```

`analyze` will detect the desired type of data analysis to perform based on whether the `ydata` argument is supplied, and whether the `xdata` argument is a two-dimensional array-like object. 

The `xdata` and `ydata` arguments can accept most python iterable objects, with the exception of strings. For example, `xdata` will accept a python list or tuple, a numpy array, or a pandas Series. Internally, iterable objects are converted to a Vector object, which is a numpy array of type `float64`.

If only the `xdata` argument is passed and it is a one-dimensional vector, the analysis performed will be a histogram of the vector with basic statistics and Shapiro-Wilk normality test. This is useful for visualizing the distribution of the vector.

If `xdata` and `ydata` are supplied and are both one-dimensional vectors, the correlation between the two vectors will be graphed and calculated. If there are non-numeric or missing values in either vector, they will be ignored. Only values that are numeric in each vector, at the same index will be included in the correlation. For example, the two following vectors will yield:

```python
In[6]: example1 = [0.2, 0.25, "nan", 0.38, 0.45, 0.6]
In[7]: example2 = [0.23, 0.27, "nan", 0.35, "nan", 0.58]
In[8]: a.analyze(example1, example2)

Correlation
-----------

Pearson Coeff:
r = 0.9932
p = 0.0068

HA: There is a significant relationship between predictor and response


Linear Regression
-----------------

slope     = 0.8704
intercept = 0.0463
R^2       = 0.9932
std err   = 0.0720
p value   = 0.0068

HA: There is a significant relationship between predictor and response
```

If `xdata` is a sequence of vectors, summary statistics will be reported for each vector. If the concatenation of each vector is normally distributed and they all have equal variance, a one-way ANOVA is performed. If the data is not normally distributed or the vectors do not have equal variance, a non-parametric Kruskal-Wallis test will be performed instead of a one-way ANOVA.

It is important to note that the vectors should be independent from one another --- that is to say, there should not be values in one vector that are derived from or some how related to a value in another vector. These dependencies can lead to weird and often unpredictable results. For example, a proper use case would be if you had a vector with measurement data and another vector (or vectors) that represent a grouping applied to the measurement data. In this case, each group should be represented by it's own vector, which are then all wrapped in a sequence. the `analyze` function accepts a `groups` argument as a list of strings of grouping names. The order of the group names should match the order of the vectors passed to `xdata`. For example:

```python
In[10]: group_a = np.random.randn(6)
In[11]: group_b = np.random.randn(7)
In[12]: group_c = np.random.randn(5)
In[13]: group_d = np.random.randn(8)
In[14]: names = ["group_a", "group_b", "group_c", "group_d"]
In[17]: data = [group_a, group_b, group_c, group_d]
In[18]: a.analyze(data, groups=names)
Count     Mean      Std.      Max       50%       Min       Group
----------------------------------------------------------------------
6         0.280     1.008     1.489     0.458     -1.555    group_a   
7         0.131     1.596     1.980     0.678     -2.058    group_b   
5         -0.300    0.932     1.061     -0.457    -1.428    group_c   
8         0.246     0.944     1.964     0.369     -1.284    group_d   

Bartlett Test
--------
T value = 2.3708
p value = 0.4991
H0: Variances are equal

ANOVA
--------
f value = 0.2832
p value = 0.8369
H0: Group means are matched
```

### What's new in sci-analysis version 1.3?

In version 1.3, sci-analysis has been re-written from scratch and is now object oriented. sci-analysis is now a python package of modules with classes instead of a single module with functions. The reason for this change is to make the code easier to follow and to establish a code base that can be more easily updated and modified in the future. The change should be mostly transparent, however, the names of individual tests have changed, and some of them now need to be called with the module name.

