
# sci-analysis

An easy to use and powerful python-based data exploration and analysis tool

## Current Version

2.1 --- Released March 30, 2018

### What is sci-analysis?

sci-analysis is a python package for quickly performing statistical data analysis. It aims to make performing Exploratory Data Analysis (EDA) easier for newcomers and experienced data analysts alike by abstracting away the specific SciPy, NumPy, and Matplotlib commands. This is accomplished by using sci-analysis's ``analyze()`` function.

With EDA, using summary statistics and tables alone can produce misleading results. Likewise, analysis of statistical graphs alone can be interpreted differently from user to user. For this reason, sci-analysis provides a graphical representation of the supplied data as well as performs the statistical analysis. 

What makes sci-analysis unique and easy to use is the ``analyze()`` function, which is smart enough to determine the correct analysis and tests to perform based on the shape of the data or number of arguments you provide, as well as how the data is distributed.

Currently, sci-analysis is capable of performing four common statistical analysis techniques:
* Histograms and summary of numeric data
* Histograms and frequency of categorical data
* Bivariate and linear regression analysis
* Location testing

### What's new in sci-analysis version 2.1?

* Version 2.1 makes improvements to Statistical output and plots.
* Tukey-Kramer circles were added to the Oneway analysis plot.
* Grand Mean and Grand Median were added to the Oneway analysis plot.
* Overall Statistics were added to Oneway analysis.
* Overall Statistics were added to Categorical analysis.
* The Categorical analysis graph was changed to improve the appearance.

### Getting started with sci-analysis

sci-analysis requires python 2.7, 3.5, or 3.6.

If one of these three version of python is already installed then this section can be skipped.

If you use MacOS or Linux, python should already be installed. You can check by opening a terminal window and typing ``which python`` on the command line. To verify what version of python you have installed, type ``python --version`` at the command line. If the version is 2.7.x, 3.5.x or 3.6.x, where x is any number, sci-analysis should work properly.

.. note:: It is not recommended to use sci-analysis with the system installed python. This is because the version of python that comes with your OS will require root permission to manage, might be changed when upgrading the OS, and can break your OS if critical packages are accidentally removed. More info on why the system python should not be used can be found here: https://github.com/MacPython/wiki/wiki/Which-Python

If you are on Windows, you might need to install python. You can check to see if python is installed by clicking the Start button, typing ``cmd`` in the run text box, then type ``python.exe`` on the command line. If you receive an error message, you need to install python. 

The easiest way to install python on any OS is by installing Anaconda or Mini-conda from this page:

<https://www.continuum.io/downloads>

If you are on MacOS and have GCC installed, python can be installed with homebrew using the command:
```
brew install python
```

If you are on Linux, python can be installed with pyenv using the instructions here:
<https://github.com/pyenv/pyenv>

If you are on Windows, you can download the python binary from the following page, but be warned that compiling the required packages will be required using this method:

<https://www.python.org/downloads/windows/>

### Installing sci-analysis

sci-analysis can be installed with pip by typing the following:

```
pip install sci_analysis
```

.. note:: The package name is ``sci_analysis`` with an underscore.

On Linux, you can install pip from your OS package manager. If you have Anaconda or Mini-conda, pip should already be installed. Otherwise, you can download pip from the following page:

<https://pypi.python.org/pypi/pip>

sci-analysis works best in conjunction with the excellent pandas and jupyter notebook python packages. If you don't have either of these packages installed, you can install them by typing the following:

```
pip install pandas
pip install jupyter
```

### Using sci-analysis

From the python interpreter or in the first cell of a Jupyter notebook, type:


```python


import numpy as np
import scipy.stats as st
from sci_analysis import analyze
```

This will tell python to import the sci-analysis function ``analyze()``.

.. note:: Alternatively, the function ``analyse()`` can be imported instead, as it is an alias for ``analyze()``. For the case of this documentation, ``analyze()`` will be used for consistency.

If you are using sci-analysis in a Jupyter notebook, you need to use the following code instead to enable inline plots:


```python
%matplotlib inline
import numpy as np
import scipy.stats as st
from sci_analysis import analyze
```

Now, sci-analysis should be ready to use. Try the following code:


```python
np.random.seed(987654321)
data = st.norm.rvs(size=1000)
analyze(data)
```


![png](./img/sci_analysis_main_19_0.png)


    
    
    Statistics
    ----------
    
    n         =  1000
    Mean      =  0.0551
    Std Dev   =  1.0287
    Std Error =  0.0325
    Skewness  = -0.1439
    Kurtosis  = -0.0931
    Maximum   =  3.4087
    75%       =  0.7763
    50%       =  0.0897
    25%       = -0.6324
    Minimum   = -3.1586
    IQR       =  1.4087
    Range     =  6.5673
    
    
    Shapiro-Wilk test for normality
    -------------------------------
    
    alpha   =  0.0500
    W value =  0.9979
    p value =  0.2591
    
    H0: Data is normally distributed
    


A histogram, box plot, summary stats, and test for normality of the data should appear above. 

.. note:: numpy and scipy.stats were only imported for the purpose of the above example. sci-analysis uses numpy and scipy internally, so it isn't necessary to import them unless you want to explicitly use them. 

A histogram and statistics for categorical data can be performed with the following command:


```python
pets = ['dog', 'cat', 'rat', 'cat', 'rabbit', 'dog', 'hamster', 'cat', 'rabbit', 'dog', 'dog']
analyze(pets)
```


![png](./img/sci_analysis_main_21_0.png)


    
    
    Overall Statistics
    ------------------
    
    Total            =  11
    Number of Groups =  5
    
    
    Statistics
    ----------
    
    Rank          Frequency     Percent       Category      
    --------------------------------------------------------
    1             4              36.3636      dog           
    2             3              27.2727      cat           
    3             2              18.1818      rabbit        
    4             1              9.0909       hamster       
    4             1              9.0909       rat           


Let's examine the ``analyze()`` function in more detail. Here's the signature for the ``analyze()`` function:


```python
from inspect import signature
print(analyze.__name__, signature(analyze))
print(analyze.__doc__)
```

    analyze (xdata, ydata=None, groups=None, alpha=0.05, **kwargs)
    
        Automatically performs a statistical analysis based on the input arguments.
    
        Parameters
        ----------
        xdata : array-like
            The primary set of data.
        ydata : array-like
            The response data set.
        groups : array-like
            The group names used for a oneway analysis.
        alpha : float
            The sensitivity to use for hypothesis tests.
    
        Returns
        -------
        xdata, ydata : tuple(array-like, array-like)
            The input xdata and ydata.
    
        Notes
        -----
        xdata : array-like, ydata : None - Distribution
        xdata : array-like, ydata : array-like -- Bivariate
        xdata : list(array-like) or dict(array-like), ydata : None -- Oneway
    
        


``analyze()`` will detect the desired type of data analysis to perform based on whether the ``ydata`` argument is supplied, and whether the ``xdata`` argument is a two-dimensional array-like object. 

The ``xdata`` and ``ydata`` arguments can accept most python array-like objects, with the exception of strings. For example, ``xdata`` will accept a python list, tuple, numpy array, or a pandas Series object. Internally, iterable objects are converted to a Vector object, which is a pandas Series of type ``float64``.

.. note:: A one-dimensional list, tuple, numpy array, or pandas Series object will all be referred to as a vector throughout the documentation.

If only the ``xdata`` argument is passed and it is a one-dimensional vector of numeric values, the analysis performed will be a histogram of the vector with basic statistics and Shapiro-Wilk normality test. This is useful for visualizing the distribution of the vector. If only the ``xdata`` argument is passed and it is a one-dimensional vector of categorical (string) values, the analysis performed will be a histogram of categories with rank, frequencies and percentages displayed.

If ``xdata`` and ``ydata`` are supplied and are both equal length one-dimensional vectors of numeric data, an x/y scatter plot with line fit will be graphed and the correlation between the two vectors will be calculated. If there are non-numeric or missing values in either vector, they will be ignored. Only values that are numeric in each vector, at the same index will be included in the correlation. For example, the two following two vectors will yield:



```python
example1 = [0.2, 0.25, 0.27, np.nan, 0.32, 0.38, 0.39, np.nan, 0.42, 0.43, 0.47, 0.51, 0.52, 0.56, 0.6]
example2 = [0.23, 0.27, 0.29, np.nan, 0.33, 0.35, 0.39, 0.42, np.nan, 0.46, 0.48, 0.49, np.nan, 0.5, 0.58]
analyze(example1, example2)
```


![png](./img/sci_analysis_main_25_0.png)


    
    
    Linear Regression
    -----------------
    
    n         =  11
    Slope     =  0.8467
    Intercept =  0.0601
    r         =  0.9836
    r^2       =  0.9674
    Std Err   =  0.0518
    p value   =  0.0000
    
    
    
    Pearson Correlation Coefficient
    -------------------------------
    
    alpha   =  0.0500
    r value =  0.9836
    p value =  0.0000
    
    HA: There is a significant relationship between predictor and response
    


If ``xdata`` is a sequence or dictionary of vectors, a location test and summary statistics for each vector will be performed. If each vector is normally distributed and they all have equal variance, a one-way ANOVA is performed. If the data is not normally distributed or the vectors do not have equal variance, a non-parametric Kruskal-Wallis test will be performed instead of a one-way ANOVA.

.. note:: Vectors should be independent from one another --- that is to say, there shouldn't be values in one vector that are derived from or some how related to a value in another vector. These dependencies can lead to weird and often unpredictable results. 

A proper use case for a location test would be if you had a table with measurement data for multiple groups, such as test scores per class, average height per country or measurements per trial run, where the classes, countries, and trials are the groups. In this case, each group should be represented by it's own vector, which are then all wrapped in a dictionary or sequence. 

If ``xdata`` is supplied as a dictionary, the keys are the names of the groups and the values are the array-like objects that represent the vectors. Alternatively, ``xdata`` can be a python sequence of the vectors and the ``groups`` argument a list of strings of the group names. The order of the group names should match the order of the vectors passed to ``xdata``. 

.. note:: Passing the data for each group into ``xdata`` as a sequence or dictionary is often referred to as "unstacked" data. With unstacked data, the values for each group are in their own vector. Alternatively, if values are in one vector and group names in another vector of equal length, this format is referred to as "stacked" data. The ``analyze()`` function can handle either stacked or unstacked data depending on which is most convenient.

For example:


```python
np.random.seed(987654321)
group_a = st.norm.rvs(size=50)
group_b = st.norm.rvs(size=25)
group_c = st.norm.rvs(size=30)
group_d = st.norm.rvs(size=40)
analyze({"Group A": group_a, "Group B": group_b, "Group C": group_c, "Group D": group_d})
```


![png](./img/sci_analysis_main_27_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  145
    Grand Mean       =  0.0598
    Pooled Std Dev   =  1.0992
    Grand Median     =  0.0741
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    50            -0.0891        1.1473       -2.4036       -0.2490        2.2466       Group A       
    25             0.2403        0.9181       -1.8853        0.3791        1.6715       Group B       
    30            -0.1282        1.0652       -2.4718       -0.0266        1.7617       Group C       
    40             0.2159        1.1629       -2.2678        0.1747        3.1400       Group D       
    
    
    Bartlett Test
    -------------
    
    alpha   =  0.0500
    T value =  1.8588
    p value =  0.6022
    
    H0: Variances are equal
    
    
    
    Oneway ANOVA
    ------------
    
    alpha   =  0.0500
    f value =  1.0813
    p value =  0.3591
    
    H0: Group means are matched
    


In the example above, sci-analysis is telling us the four groups are normally distributed (by use of the Bartlett Test, Oneway ANOVA and the near straight line fit on the quantile plot), the groups have equal variance and the groups have matching means. The only significant difference between the four groups is the sample size we specified. Let's try another example, but this time change the variance of group B:


```python
np.random.seed(987654321)
group_a = st.norm.rvs(0.0, 1, size=50)
group_b = st.norm.rvs(0.0, 3, size=25)
group_c = st.norm.rvs(0.1, 1, size=30)
group_d = st.norm.rvs(0.0, 1, size=40)
analyze({"Group A": group_a, "Group B": group_b, "Group C": group_c, "Group D": group_d})
```


![png](./img/sci_analysis_main_29_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  145
    Grand Mean       =  0.2049
    Pooled Std Dev   =  1.5350
    Grand Median     =  0.1241
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    50            -0.0891        1.1473       -2.4036       -0.2490        2.2466       Group A       
    25             0.7209        2.7543       -5.6558        1.1374        5.0146       Group B       
    30            -0.0282        1.0652       -2.3718        0.0734        1.8617       Group C       
    40             0.2159        1.1629       -2.2678        0.1747        3.1400       Group D       
    
    
    Bartlett Test
    -------------
    
    alpha   =  0.0500
    T value =  42.7597
    p value =  0.0000
    
    HA: Variances are not equal
    
    
    
    Kruskal-Wallis
    --------------
    
    alpha   =  0.0500
    h value =  7.1942
    p value =  0.0660
    
    H0: Group means are matched
    


In the example above, group B has a standard deviation of 2.75 compared to the other groups that are approximately 1. The quantile plot on the right also shows group B has a much steeper slope compared to the other groups, implying a larger variance. Also, the Kruskal-Wallis test was used instead of the Oneway ANOVA because the pre-requisite of equal variance was not met.

In another example, let's compare groups that have different distributions and different means:


```python
np.random.seed(987654321)
group_a = st.norm.rvs(0.0, 1, size=50)
group_b = st.norm.rvs(0.0, 3, size=25)
group_c = st.weibull_max.rvs(1.2, size=30)
group_d = st.norm.rvs(0.0, 1, size=40)
analyze({"Group A": group_a, "Group B": group_b, "Group C": group_c, "Group D": group_d})
```


![png](./img/sci_analysis_main_31_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  145
    Grand Mean       = -0.0694
    Pooled Std Dev   =  1.4903
    Grand Median     = -0.1148
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    50            -0.0891        1.1473       -2.4036       -0.2490        2.2466       Group A       
    25             0.7209        2.7543       -5.6558        1.1374        5.0146       Group B       
    30            -1.0340        0.8029       -2.7632       -0.7856       -0.0606       Group C       
    40             0.1246        1.1081       -1.9334        0.0193        3.1400       Group D       
    
    
    Levene Test
    -----------
    
    alpha   =  0.0500
    W value =  10.1675
    p value =  0.0000
    
    HA: Variances are not equal
    
    
    
    Kruskal-Wallis
    --------------
    
    alpha   =  0.0500
    h value =  23.8694
    p value =  0.0000
    
    HA: Group means are not matched
    


The above example models group C as a Weibull distribution, while the other groups are normally distributed. You can see the difference in the distributions by the one-sided tail on the group C boxplot, and the curved shape of group C on the quantile plot. Group C also has significantly the lowest mean as indicated by the Tukey-Kramer circles and the Kruskal-Wallis test.

### Using sci-analysis with pandas

Pandas is a python package that simplifies working with tabular or relational data. Because columns and rows of data in a pandas DataFrame are naturally array-like, using pandas with sci-analysis is the preferred way to use sci-analysis.

Let's create a pandas DataFrame to use for analysis:


```python
import pandas as pd
np.random.seed(987654321)
df = pd.DataFrame({'One'   : st.norm.rvs(0.0, 1, size=60),
                   'Two'   : st.norm.rvs(0.0, 3, size=60),
                   'Three' : st.weibull_max.rvs(1.2, size=60),
                   'Four'  : st.norm.rvs(0.0, 1, size=60),
                   'Month' : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] * 5,
                   'Condition' : ['Group A', 'Group B', 'Group C', 'Group D'] * 15})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Condition</th>
      <th>Four</th>
      <th>Month</th>
      <th>One</th>
      <th>Three</th>
      <th>Two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Group A</td>
      <td>0.058648</td>
      <td>Jan</td>
      <td>2.246551</td>
      <td>-1.773298</td>
      <td>3.067338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Group B</td>
      <td>0.882809</td>
      <td>Feb</td>
      <td>-0.645918</td>
      <td>-1.507548</td>
      <td>4.390899</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Group C</td>
      <td>-0.393765</td>
      <td>Mar</td>
      <td>-1.183577</td>
      <td>-1.098097</td>
      <td>4.125750</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Group D</td>
      <td>-1.115985</td>
      <td>Apr</td>
      <td>1.929493</td>
      <td>-1.484530</td>
      <td>-5.655830</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Group A</td>
      <td>0.201746</td>
      <td>May</td>
      <td>0.066814</td>
      <td>-0.695673</td>
      <td>1.137421</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Group B</td>
      <td>-1.739743</td>
      <td>Jun</td>
      <td>-0.997060</td>
      <td>-0.718185</td>
      <td>1.281548</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Group C</td>
      <td>-0.088919</td>
      <td>Jul</td>
      <td>0.815330</td>
      <td>-0.784834</td>
      <td>5.014646</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Group D</td>
      <td>-0.813064</td>
      <td>Aug</td>
      <td>-1.340937</td>
      <td>-0.214690</td>
      <td>2.800435</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Group A</td>
      <td>-0.605264</td>
      <td>Sep</td>
      <td>1.910637</td>
      <td>-0.345766</td>
      <td>-2.959447</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Group B</td>
      <td>-0.198116</td>
      <td>Oct</td>
      <td>0.709583</td>
      <td>-0.907204</td>
      <td>-0.363670</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Group C</td>
      <td>1.379706</td>
      <td>Nov</td>
      <td>1.129347</td>
      <td>-1.081587</td>
      <td>-0.365354</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Group D</td>
      <td>-1.020806</td>
      <td>Dec</td>
      <td>-1.139548</td>
      <td>-0.459009</td>
      <td>-3.910547</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Group A</td>
      <td>0.128327</td>
      <td>Jan</td>
      <td>0.309055</td>
      <td>-1.522231</td>
      <td>0.502254</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Group B</td>
      <td>-0.281124</td>
      <td>Feb</td>
      <td>-0.594467</td>
      <td>-0.449248</td>
      <td>1.138284</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Group C</td>
      <td>1.319009</td>
      <td>Mar</td>
      <td>-1.091333</td>
      <td>-0.552761</td>
      <td>4.585332</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Group D</td>
      <td>-0.547517</td>
      <td>Apr</td>
      <td>0.366702</td>
      <td>-0.298619</td>
      <td>1.737304</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Group A</td>
      <td>1.638618</td>
      <td>May</td>
      <td>-0.387678</td>
      <td>-0.869620</td>
      <td>-1.536255</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Group B</td>
      <td>-0.389347</td>
      <td>Jun</td>
      <td>-1.637016</td>
      <td>-0.926374</td>
      <td>0.987896</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Group C</td>
      <td>0.018815</td>
      <td>Jul</td>
      <td>1.725978</td>
      <td>-2.258327</td>
      <td>5.285016</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Group D</td>
      <td>-1.519705</td>
      <td>Aug</td>
      <td>0.785090</td>
      <td>-0.099053</td>
      <td>-1.174279</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Group A</td>
      <td>0.246948</td>
      <td>Sep</td>
      <td>-0.836660</td>
      <td>-0.255706</td>
      <td>-0.460675</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Group B</td>
      <td>-0.760512</td>
      <td>Oct</td>
      <td>0.486172</td>
      <td>-0.159479</td>
      <td>-2.296232</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Group C</td>
      <td>0.297368</td>
      <td>Nov</td>
      <td>0.819392</td>
      <td>-0.635424</td>
      <td>-5.949174</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Group D</td>
      <td>-0.986769</td>
      <td>Dec</td>
      <td>1.556978</td>
      <td>-0.566483</td>
      <td>-0.053715</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Group A</td>
      <td>1.205519</td>
      <td>Jan</td>
      <td>0.483678</td>
      <td>-2.557816</td>
      <td>-2.316271</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Group B</td>
      <td>0.423745</td>
      <td>Feb</td>
      <td>-1.832651</td>
      <td>-0.727140</td>
      <td>-3.053756</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Group C</td>
      <td>-1.674542</td>
      <td>Mar</td>
      <td>-1.083556</td>
      <td>-0.901519</td>
      <td>5.165781</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Group D</td>
      <td>1.109368</td>
      <td>Apr</td>
      <td>-1.332366</td>
      <td>-0.659552</td>
      <td>-2.580036</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Group A</td>
      <td>-0.845824</td>
      <td>May</td>
      <td>0.441650</td>
      <td>-0.335586</td>
      <td>1.324607</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Group B</td>
      <td>0.221365</td>
      <td>Jun</td>
      <td>2.093856</td>
      <td>-0.656918</td>
      <td>1.813717</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Group C</td>
      <td>-1.075979</td>
      <td>Jul</td>
      <td>-1.432219</td>
      <td>-1.286806</td>
      <td>4.906245</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Group D</td>
      <td>0.055312</td>
      <td>Aug</td>
      <td>0.988241</td>
      <td>-2.928589</td>
      <td>1.871635</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Group A</td>
      <td>1.595664</td>
      <td>Sep</td>
      <td>-0.297481</td>
      <td>-0.684245</td>
      <td>-1.049538</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Group B</td>
      <td>-0.624568</td>
      <td>Oct</td>
      <td>-0.294577</td>
      <td>-0.629722</td>
      <td>0.291384</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Group C</td>
      <td>-1.882894</td>
      <td>Nov</td>
      <td>-1.199973</td>
      <td>-0.202688</td>
      <td>-0.051015</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Group D</td>
      <td>-1.238447</td>
      <td>Dec</td>
      <td>-0.142682</td>
      <td>-2.619512</td>
      <td>3.522920</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Group A</td>
      <td>0.580960</td>
      <td>Jan</td>
      <td>-1.746777</td>
      <td>-0.219328</td>
      <td>-7.415294</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Group B</td>
      <td>-0.094435</td>
      <td>Feb</td>
      <td>-0.437626</td>
      <td>-0.141427</td>
      <td>0.805884</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Group C</td>
      <td>0.821534</td>
      <td>Mar</td>
      <td>-1.205166</td>
      <td>-0.493761</td>
      <td>-0.105672</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Group D</td>
      <td>1.428968</td>
      <td>Apr</td>
      <td>-0.610066</td>
      <td>-0.397653</td>
      <td>-1.842630</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Group A</td>
      <td>-0.890143</td>
      <td>May</td>
      <td>-0.203453</td>
      <td>-0.151268</td>
      <td>2.323542</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Group B</td>
      <td>0.437110</td>
      <td>Jun</td>
      <td>0.085310</td>
      <td>-0.002376</td>
      <td>-2.053241</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Group C</td>
      <td>1.141289</td>
      <td>Jul</td>
      <td>1.853726</td>
      <td>-1.035302</td>
      <td>-5.176661</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Group D</td>
      <td>-1.391369</td>
      <td>Aug</td>
      <td>-0.614827</td>
      <td>-0.626213</td>
      <td>1.266392</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Group A</td>
      <td>-1.145177</td>
      <td>Sep</td>
      <td>0.091151</td>
      <td>-0.054092</td>
      <td>-5.721601</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Group B</td>
      <td>-0.880138</td>
      <td>Oct</td>
      <td>0.367027</td>
      <td>-2.530850</td>
      <td>1.929861</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Group C</td>
      <td>-0.487270</td>
      <td>Nov</td>
      <td>0.337290</td>
      <td>-1.736722</td>
      <td>3.160379</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Group D</td>
      <td>0.416363</td>
      <td>Dec</td>
      <td>-2.403575</td>
      <td>-0.879869</td>
      <td>1.846666</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Group A</td>
      <td>-0.151296</td>
      <td>Jan</td>
      <td>-1.008465</td>
      <td>-0.556609</td>
      <td>-0.148862</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Group B</td>
      <td>0.645674</td>
      <td>Feb</td>
      <td>-0.352184</td>
      <td>-1.424446</td>
      <td>5.654610</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Group C</td>
      <td>1.014516</td>
      <td>Mar</td>
      <td>0.598030</td>
      <td>-1.804494</td>
      <td>-2.689915</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Group D</td>
      <td>-1.349427</td>
      <td>Apr</td>
      <td>0.573027</td>
      <td>-1.235840</td>
      <td>1.372438</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Group A</td>
      <td>0.513536</td>
      <td>May</td>
      <td>0.215652</td>
      <td>-1.683723</td>
      <td>4.065310</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Group B</td>
      <td>-2.234853</td>
      <td>Jun</td>
      <td>-0.436220</td>
      <td>-1.280547</td>
      <td>-2.193357</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Group C</td>
      <td>1.269432</td>
      <td>Jul</td>
      <td>0.209028</td>
      <td>-1.326379</td>
      <td>-0.720595</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Group D</td>
      <td>-0.807231</td>
      <td>Aug</td>
      <td>0.406017</td>
      <td>-0.503970</td>
      <td>0.221715</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Group A</td>
      <td>0.917287</td>
      <td>Sep</td>
      <td>-0.112333</td>
      <td>-1.935414</td>
      <td>3.377393</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Group B</td>
      <td>-0.406160</td>
      <td>Oct</td>
      <td>0.341139</td>
      <td>-0.292610</td>
      <td>2.775356</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Group C</td>
      <td>0.707101</td>
      <td>Nov</td>
      <td>-1.435777</td>
      <td>-0.330216</td>
      <td>-3.183457</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Group D</td>
      <td>0.927546</td>
      <td>Dec</td>
      <td>0.719249</td>
      <td>-0.388752</td>
      <td>-2.281663</td>
    </tr>
  </tbody>
</table>
</div>



This creates a table (pandas DataFrame object) with 6 columns and an index which is the row id. The following command can be used to analyze the distribution of the column titled **One**:


```python
analyze(df['One'], 
        name='Column One', 
        title='Distribution from pandas')
```


![png](./img/sci_analysis_main_37_0.png)


    
    
    Statistics
    ----------
    
    n         =  60
    Mean      = -0.0562
    Std Dev   =  1.0779
    Std Error =  0.1392
    Skewness  =  0.1879
    Kurtosis  = -0.5109
    Maximum   =  2.2466
    75%       =  0.5793
    50%       = -0.0228
    25%       = -0.9999
    Minimum   = -2.4036
    IQR       =  1.5792
    Range     =  4.6501
    
    
    Shapiro-Wilk test for normality
    -------------------------------
    
    alpha   =  0.0500
    W value =  0.9796
    p value =  0.4131
    
    H0: Data is normally distributed
    


Anywhere you use a python list or numpy Array in sci-analysis, you can use a column or row of a pandas DataFrame (known in pandas terms as a Series). This is because a pandas Series has much of the same behavior as a numpy Array, causing sci-analysis to handle a pandas Series as if it were a numpy Array.

By passing two array-like arguments to the ``analyze()`` function, the correlation can be determined between the two array-like arguments. The following command can be used to analyze the correlation between columns **One** and **Three**:


```python
analyze(df['One'], df['Three'], 
        xname='Column One', 
        yname='Column Three', 
        title='Bivariate Analysis between Column One and Column Three')
```


![png](./img/sci_analysis_main_39_0.png)


    
    
    Linear Regression
    -----------------
    
    n         =  60
    Slope     = -0.1807
    Intercept = -0.9249
    r         = -0.2742
    r^2       =  0.0752
    Std Err   =  0.0832
    p value   =  0.0340
    
    
    
    Spearman Correlation Coefficient
    --------------------------------
    
    alpha   =  0.0500
    r value = -0.2289
    p value =  0.0785
    
    H0: There is no significant relationship between predictor and response
    


Since there isn't a correlation between columns **One** and **Three**, it might be useful to see where most of the data is concentrated. This can be done by adding the argument ``contours=True`` and turning off the best fit line with ``fit=False``. For example:


```python
analyze(df['One'], df['Three'], 
        xname='Column One', 
        yname='Column Three',
        contours=True,
        fit=False,
        title='Bivariate Analysis between Column One and Column Three')
```


![png](./img/sci_analysis_main_41_0.png)


    
    
    Linear Regression
    -----------------
    
    n         =  60
    Slope     = -0.1807
    Intercept = -0.9249
    r         = -0.2742
    r^2       =  0.0752
    Std Err   =  0.0832
    p value   =  0.0340
    
    
    
    Spearman Correlation Coefficient
    --------------------------------
    
    alpha   =  0.0500
    r value = -0.2289
    p value =  0.0785
    
    H0: There is no significant relationship between predictor and response
    


To check whether an individual **Condition** correlates between Column **One** and Column **Three**, the same analysis can be done, but this time by passing the **Condition** column to the groups argument. For example:


```python
analyze(df['One'], df['Three'],
        xname='Column One',
        yname='Column Three',
        groups=df['Condition'],
        title='Bivariate Analysis between Column One and Column Three')
```


![png](./img/sci_analysis_main_43_0.png)


    
    
    Linear Regression
    -----------------
    
    n             Slope         Intercept     r^2           Std Err       p value       Group         
    --------------------------------------------------------------------------------------------------
    15            -0.2738       -0.8880        0.1246        0.2012        0.1968       Group A       
    15             0.0303       -0.8172        0.0020        0.1883        0.8745       Group B       
    15            -0.2615       -1.0552        0.2950        0.1121        0.0364       Group C       
    15            -0.1697       -0.8938        0.0578        0.1900        0.3879       Group D       
    
    
    Spearman Correlation Coefficient
    --------------------------------
    
    n             r value       p value       Group         
    --------------------------------------------------------
    15            -0.4107        0.1283       Group A       
    15             0.1857        0.5075       Group B       
    15            -0.4500        0.0924       Group C       
    15            -0.1679        0.5499       Group D       


The borders of the graph have boxplots for all the data points on the x-axis and y-axis, regardless of which group they belong to. The borders can be removed by adding the argument ``boxplot_borders=False``.

According to the Spearman Correlation, there is no significant correlation among the groups. Group C appears to be the closest to showing a significant correlation, but it can be difficult to see the data points for Group C with so many colors on the graph. The Group C data points can be highlighted by using the argument ``highlight=['Group C']``. In fact, any number of groups can be highlighted by passing a list of the group names using the ``highlight`` argument.


```python
analyze(df['One'], df['Three'],
        xname='Column One',
        yname='Column Three',
        groups=df['Condition'],
        boxplot_borders=False,
        highlight=['Group C'],
        title='Bivariate Analysis between Column One and Column Three')
```


![png](./img/sci_analysis_main_45_0.png)


    
    
    Linear Regression
    -----------------
    
    n             Slope         Intercept     r^2           Std Err       p value       Group         
    --------------------------------------------------------------------------------------------------
    15            -0.2738       -0.8880        0.1246        0.2012        0.1968       Group A       
    15             0.0303       -0.8172        0.0020        0.1883        0.8745       Group B       
    15            -0.2615       -1.0552        0.2950        0.1121        0.0364       Group C       
    15            -0.1697       -0.8938        0.0578        0.1900        0.3879       Group D       
    
    
    Spearman Correlation Coefficient
    --------------------------------
    
    n             r value       p value       Group         
    --------------------------------------------------------
    15            -0.4107        0.1283       Group A       
    15             0.1857        0.5075       Group B       
    15            -0.4500        0.0924       Group C       
    15            -0.1679        0.5499       Group D       


Performing a location test on data in a pandas DataFrame requires some explanation. A location test can be performed with stacked or unstacked data. One method will be easier than the other depending on how the data to be analyzed is stored. In the example DataFrame used so far, to perform a location test between the groups in the **Condition** column, the stacked method will be easier to use. 

Let's start with an example. The following code will perform a location test using each of the four values in the **Condition** column:


```python
analyze(df['Two'], 
        groups=df['Condition'],
        categories='Condition',
        name='Column Two',
        title='Oneway from pandas')
```


![png](./img/sci_analysis_main_47_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  60
    Grand Mean       =  0.2740
    Pooled Std Dev   =  3.1385
    Grand Median     =  0.0853
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    15            -0.3873        3.2660       -7.4153       -0.1489        4.0653       Group A       
    15             0.7406        2.4806       -3.0538        0.9879        5.6546       Group B       
    15             0.9334        3.9554       -5.9492       -0.0510        5.2850       Group C       
    15            -0.1906        2.6335       -5.6558        0.2217        3.5229       Group D       
    
    
    Bartlett Test
    -------------
    
    alpha   =  0.0500
    T value =  3.7931
    p value =  0.2847
    
    H0: Variances are equal
    
    
    
    Oneway ANOVA
    ------------
    
    alpha   =  0.0500
    f value =  0.6628
    p value =  0.5784
    
    H0: Group means are matched
    


From the graph, there are four groups: Group A, Group B, Group C and Group D in Column **Two**. The analysis shows that the variances are equal and there is no significant difference in the means. Noting the tests that are being performed, the Bartlett test is being used to check for equal variance because all four groups are normally distributed, and the Oneway ANOVA is being used to test if all means are equal because all four groups are normally distributed and the variances are equal. However, if not all the groups are normally distributed, the Levene Test will be used to check for equal variance instead of the Bartlett Test. Also, if the groups are not normally distributed or the variances are not equal, the Kruskal-Wallis test will be used instead of the Oneway ANOVA.

If instead the four columns **One**, **Two**, **Three** and **Four** are to be analyzed, the easier way to perform the analysis is with the unstacked method. The following code will perform a location test of the four columns:


```python
analyze([df['One'], df['Two'], df['Three'], df['Four']], 
        groups=['One', 'Two', 'Three', 'Four'],
        categories='Columns',
        title='Unstacked Oneway')
```


![png](./img/sci_analysis_main_49_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  240
    Grand Mean       = -0.1994
    Pooled Std Dev   =  1.7536
    Grand Median     = -0.0572
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    60            -0.1006        0.9761       -2.2349       -0.0917        1.6386       Four          
    60            -0.0562        1.0779       -2.4036       -0.0228        2.2466       One           
    60            -0.9148        0.7104       -2.9286       -0.6900       -0.0024       Three         
    60             0.2740        3.1115       -7.4153        0.3968        5.6546       Two           
    
    
    Levene Test
    -----------
    
    alpha   =  0.0500
    W value =  44.7814
    p value =  0.0000
    
    HA: Variances are not equal
    
    
    
    Kruskal-Wallis
    --------------
    
    alpha   =  0.0500
    h value =  24.0538
    p value =  0.0000
    
    HA: Group means are not matched
    


To perform a location test using the unstacked method, the columns to be analyzed are passed in a list or tuple, and the groups argument needs to be a list or tuple of the group names. One thing to note is that the groups argument was used to explicitly define the group names. This will only work if the group names and order are known in advance. If they are unknown, a dictionary comprehension can be used instead of a list comprehension to to get the group names along with the data:


```python
analyze({'One': df['One'], 'Two': df['Two'], 'Three': df['Three'], 'Four': df['Four']}, 
        categories='Columns',
        title='Unstacked Oneway Using a Dictionary Comp')
```


![png](./img/sci_analysis_main_51_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  240
    Grand Mean       = -0.1994
    Pooled Std Dev   =  1.7536
    Grand Median     = -0.0572
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    60            -0.1006        0.9761       -2.2349       -0.0917        1.6386       Four          
    60            -0.0562        1.0779       -2.4036       -0.0228        2.2466       One           
    60            -0.9148        0.7104       -2.9286       -0.6900       -0.0024       Three         
    60             0.2740        3.1115       -7.4153        0.3968        5.6546       Two           
    
    
    Levene Test
    -----------
    
    alpha   =  0.0500
    W value =  44.7814
    p value =  0.0000
    
    HA: Variances are not equal
    
    
    
    Kruskal-Wallis
    --------------
    
    alpha   =  0.0500
    h value =  24.0538
    p value =  0.0000
    
    HA: Group means are not matched
    


The output will be identical to the previous example. The analysis also shows that the variances are not equal, and the means are not matched. Also, because the data in column **Three** is not normally distributed, the Levene Test is used to test for equal variance instead of the Bartlett Test, and the Kruskal-Wallis Test is used instead of the Oneway ANOVA.

With pandas, it's possible to perform advanced aggregation and filtering functions using the GroupBy object's ``apply()`` method. Since the sample sizes were small for each month in the above examples, it might be helpful to group the data by annual quarters instead. First, let's create a function that adds a column called **Quarter** to the DataFrame where the value is either Q1, Q2, Q3 or Q4 depending on the month.


```python
def set_quarter(data):
    month = data['Month']
    if month.all() in ('Jan', 'Feb', 'Mar'):
        quarter = 'Q1'
    elif month.all() in ('Apr', 'May', 'Jun'):
        quarter = 'Q2'
    elif month.all() in ('Jul', 'Aug', 'Sep'):
        quarter = 'Q3'
    elif month.all() in ('Oct', 'Nov', 'Dec'):
        quarter = 'Q4'
    else:
        quarter = 'Unknown'
    data.loc[:, 'Quarter'] = quarter
    return data
```

This function will take a GroupBy object called *data*, where *data*'s DataFrame object was grouped by month, and set the variable *quarter* based off the month. Then, a new column called **Quarter** is added to *data* where the value of each row is equal to *quarter*. Finally, the resulting DataFrame object is returned.

Using the new function is simple. The same techniques from previous examples are used, but this time, a new DataFrame object called *df2* is created by first grouping by the **Month** column then calling the ``apply()`` method which will run the ``set_quarter()`` function.


```python
quarters = ('Q1', 'Q2', 'Q3', 'Q4')
df2 = df.groupby(df['Month']).apply(set_quarter)
data = {quarter: data['Two'] for quarter, data in df2.groupby(df2['Quarter'])}
analyze([data[quarter] for quarter in quarters],
        groups=quarters,
        categories='Quarters',
        name='Column Two',
        title='Oneway of Annual Quarters')
```


![png](./img/sci_analysis_main_55_0.png)


    
    
    Overall Statistics
    ------------------
    
    Number of Groups =  4
    Total            =  60
    Grand Mean       =  0.2740
    Pooled Std Dev   =  3.1561
    Grand Median     =  0.5138
    
    
    Group Statistics
    ----------------
    
    n             Mean          Std Dev       Min           Median        Max           Group         
    --------------------------------------------------------------------------------------------------
    15             0.9138        3.7034       -7.4153        0.8059        5.6546       Q1            
    15             0.0122        2.5243       -5.6558        1.1374        4.0653       Q2            
    15             0.4987        3.4778       -5.7216        0.2217        5.2850       Q3            
    15            -0.3286        2.7681       -5.9492       -0.0537        3.5229       Q4            
    
    
    Bartlett Test
    -------------
    
    alpha   =  0.0500
    T value =  2.6544
    p value =  0.4480
    
    H0: Variances are equal
    
    
    
    Oneway ANOVA
    ------------
    
    alpha   =  0.0500
    f value =  0.4474
    p value =  0.7201
    
    H0: Group means are matched
    

