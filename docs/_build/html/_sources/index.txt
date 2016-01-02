.. sci_analysis documentation master file, created by
   sphinx-quickstart on Wed Dec 30 21:49:27 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


What is sci_analysis?
=====================

sci_analysis is a python module for performing rapid statistical data analysis. It provides a graphical representation of the supplied data as well as the statistical analysis. sci-analysis is smart enough to determine the correct analysis and tests to perform based on the shape of the data you provide, as well as whether the data is normally distributed.

Currently, sci_analysis can only be used for analyzing numeric data. Categorical data analysis is planned for a future version. The three types of analysis that can be performed are histograms of single vectors, correlation between two vectors and one-way ANOVA.

What's new in sci_analysis version 1.3?
=======================================

In version 1.3, sci_analysis has been re-written from scratch and is now object oriented. sci-analysis is now a python package of modules with classes instead of a single module with functions. The reason for this change is to make the code easier to follow and to establish a code base that can be easily updated and modified in the future. The change should be mostly transparent. However, the names of individual tests have changed, and some of them now need to be called with the module name.

Getting started with sci_analysis
=================================

sci_analysis require python 2.7. It has not been tested with python 3.0 or above. 

If you use OS X or Linux, python should already be installed. You can check by opening a terminal window and typing ``which python`` on the command line. To verify what version of python you have installed, type ``python --version`` at the command line. If the version is 2.7.x, where x is any number, sci_analysis should work properly.

If you are on Windows, you might need to install python. You can check to see if python is installed by clicking the Start button, typing ``cmd`` in the run text box, then type ``python.exe`` on the command line. If you receive an error message, you need to install python. You can download python from the following page:

`<https://www.python.org/downloads/windows/>`_

Installing sci_analysis
=======================

sci_analysis can be installed with pip by tying the following:::

	pip install sci_analysis
	
On Linux, you can install pip from your OS package manager. Otherwise, you can download pip from the following page:

`<https://pypi.python.org/pypi/pip>`_

sci_analysis is also compatible with pandas and works best in the iPython Notebook.

Using sci_analysis
==================

From the python interpreter, type:::

	import sci_analysis as a
	import numpy as np

**Note:** The package name is ``sci_analysis`` with an underscore.

This will tell python to import sci_analysis to your project as the object ``a``.

If you are using the iPython Notebook, you will also want to use the following code instead to enable inline plots:::

	%matplotlib inline
	import sci_analysis as a
	import numpy as np

Now, sci-analysis should be ready to use. Try the following code:::

	data = np.random.randn(100)
	a.analyze(data)

A histogram and box plot of the data should appear, as well as printed output similar to that below:::

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

If ``data`` contains missing values or strings, they will be ignored when generating the statistics and graphing the histogram.

You should probably note that numpy was only imported for the purpose of the above example. sci_analysis uses numpy internally, so it isn't necessary to import it unless you want to explicitly use it. 

Let's examine the ``analyze`` function in more detail. Here's the signature for the ``analyze`` function:::

	def analyze(xdata, ydata=None, groups=None, name=None, xname=None, yname=None, alpha=0.05, categories='Categories'):

``analyze`` will detect the desired type of data analysis to perform based on whether the ``ydata`` argument is supplied, and whether the ``xdata`` argument is a two-dimensional array-like object. 

The ``xdata`` and ``ydata`` arguments can accept most python iterable objects, with the exception of strings. For example, ``xdata`` will accept a python list or tuple, a numpy array, or a pandas Series object. Internally, iterable objects are converted to a Vector object, which is a numpy array of type ``float64``.

If only the ``xdata`` argument is passed and it is a one-dimensional vector, the analysis performed will be a histogram of the vector with basic statistics and Shapiro-Wilk normality test. This is useful for visualizing the distribution of the vector.

If ``xdata`` and ``ydata`` are supplied and are both one-dimensional vectors, an x, y scatter plot with line fit will be graphed and the correlation between the two vectors will be calculated. If there are non-numeric or missing values in either vector, they will be ignored. Only values that are numeric in each vector, at the same index will be included in the correlation. For example, the two following vectors will yield:::

	In[6]: example1 = [0.2, 0.25, "nan", 0.38, 0.45, 0.6]
	In[7]: example2 = [0.23, 0.27, "nan", 0.35, "nan", 0.58]
	In[8]: a.analyze(example1, example2)


	Linear Regression
	-----------------

	count     = 4
	slope     = 0.8704
	intercept = 0.0463
	R^2       = 0.9932
	std err   = 0.0720
	p value   = 0.0068

	HA: There is a significant relationship between predictor and response

	Correlation
	-----------

	Pearson Coeff:
	r = 0.9932
	p = 0.0068

	HA: There is a significant relationship between predictor and response

If ``xdata`` is a sequence or dictionary of vectors, summary statistics will be reported for each vector. If the concatenation of each vector is normally distributed and they all have equal variance, a one-way ANOVA is performed. If the data is not normally distributed or the vectors do not have equal variance, a non-parametric Kruskal-Wallis test will be performed instead of a one-way ANOVA.

It is important to note that the vectors should be independent from one another --- that is to say, there should not be values in one vector that are derived from or some how related to a value in another vector. These dependencies can lead to weird and often unpredictable results. 

For example, a proper use case would be if you had a table with measurement data for multiple groups, such as trial numbers or patients. In this case, each group should be represented by it's own vector, which are then all wrapped in a dictionary or sequence. 

If ``xdata`` is supplied as a dictionary, the keys are the names of the groups and the values are the iterable objects that represent the vectors. Alternatively, ``xdata`` can be a python sequence of the vectors and the ``groups`` argument a list of strings of the group names. The order of the group names should match the order of the vectors passed to ``xdata``. For example:::

	In[5]: group_a = np.random.randn(6)
	In[6]: group_b = np.random.randn(7)
	In[7]: group_c = np.random.randn(5)
	In[8]: group_d = np.random.randn(8)
	In[9]: a.analyze({"Group A": group_a, "Group B": group_b, "Group C": group_c, "Group D": group_d})
	Count       Mean        Std.        Min         Q2          Max         Group       
	------------------------------------------------------------------------------------
	8          -0.53665     0.84271    -1.30249    -0.79383     1.31658     Group D     
	7          -0.24336     1.09071    -1.69316     0.18019     1.21020     Group B     
	5           0.73371     0.95148    -0.55325     0.43994     1.70520     Group C     
	6           0.40363     1.52694    -2.16493     0.32231     2.32542     Group A     

	Bartlett Test
	-------------

	T value = 2.1667
	p value = 0.5385

	H0: Variances are equal


	Oneway ANOVA
	------------

	f value = 1.7103
	p value = 0.1941

	H0: Group means are matched
