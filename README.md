# sci-analysis
A light weight python data exploration and analysis tool by Chris Morrow

## Current Version:
1.2 --- Released May 26, 2015

### What is sci_analysis?
Sci_analysis is a python module for performing rapid statistical data analysis. It provides a graphical representation of the supplied data as well as the statistical analysis. Sci_analysis is smart enough to determine the correct analysis and tests to perform based on the shape of the data you provide, as well as whether the data is normally distributed.

Currently, sci_analysis can only be used for analyzing numeric data. Categorical data analysis is planned for a future version. The three types of analysis that can be performed are histograms of single vectors, correlation between two vectors and one-way ANOVA.

### How do I use sci_analysis?
Before using sci_analysis, be sure the following three packages are installed:
	- numpy
	- scipy
	- matplotlib
	
Sci_analysis is also compatible with pandas and works best in the iPython Notebook.

First, download to your PC or clone the repo at: 
https://github.com/cmmorrow/sci-analysis

Next, add the sci_analysis directory to your project with:

```python
sys.path.extend(['<path to directory>/sci_analysis'])
import scianalysis as a
import numpy as np
```

This will tell python were to find sci_analysis and import it to your project as the object `a`. 

If you are using the iPython Notebook, you will also want to use the following code instead to enable inline plots:

```python
%matplotlib inline
import matplotlib
sys.path.extend(['<path to directory>/sci_analysis'])
import scianalysis as a
import numpy as np
```

Now, sci_analysis should be ready to use. Try the following code:

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

You should probably note that numpy was only imported for the purpose of the above example. Sci_analysis uses numpy internally, so it isn't necessary to import it unless you want to explicitly use it. Sci_analysis can work with regular python sequences as in the following:

```
In[6]: a.clean([6, 9, 12, 15])
Out[6]: array([ 6,  9, 12, 15])

In[7]: a.clean((4, 8, 12, 16, 20))
Out[7]: array([ 4,  8, 12, 16, 20])
```

Sci_analysis is also compatible with pandas Series and DataFrame objects. To use pandas with sci_analysis, be sure to import it to your project with:

```python
import pandas as pd
```

The sci_analysis helper functions can accept a pandas Series object and return a Series as in the example below:

```
In[9]: a.clean(pd.Series([6, 9, 12, 15]))
Out[9]: 
0     6
1     9
2    12
3    15
dtype: int64
```