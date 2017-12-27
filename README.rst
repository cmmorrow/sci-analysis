============
sci-analysis
============

An easy to use and powerful python-based data exploration and analysis tool

---------------
Current Version
---------------

2.0 --- Released December 27, 2017

.. image:: https://img.shields.io/pypi/v/sci_analysis.svg   :target: https://pypi.python.org/pypi/sci_analysis
.. image:: https://img.shields.io/pypi/format/sci_analysis.svg   :target: https://pypi.python.org/pypi/sci_analysis
.. image:: https://img.shields.io/pypi/pyversions/sci_analysis.svg   :target: https://pypi.python.org/pypi/sci_analysis
.. image:: https://travis-ci.org/cmmorrow/sci-analysis.svg?branch=2.0.0   :target: https://travis-ci.org/cmmorrow/sci-analysis
.. image:: https://coveralls.io/repos/github/cmmorrow/sci-analysis/badge.svg?branch=2.0.0   :target: https://coveralls.io/github/cmmorrow/sci-analysis?branch=2.0.0

What is sci-analysis?
---------------------

sci-analysis is a python package for quickly performing statistical data analysis. It provides a graphical representation of the supplied data as well as the statistical analysis. sci-analysis is smart enough to determine the correct analysis and tests to perform based on the shape of the data you provide, as well as how the data is distributed.

The types of analysis that can be performed are histograms of numeric or categorical data, bivariate analysis of two numeric vectors, and one-way analysis of variance.

What's new in sci-analysis version 2.0?
---------------------------------------

* In version 2.0, the code base was re-factored to use pandas as the internal data structure instead of numpy. This change shouldn't have a noticeable effect, but should lead to faster releases moving forward. 
* Categorical data is now supported by passing in a single array of string values to the ``analyze`` function. 
* Multiple scatter plots can now be shown on the same graph by passing in a *groups* argument.
* Oneway analysis can now be performed on stacked data by passing in all the data to the ``analyze`` function and an array of the corresponding groups using the *groups* argument.
* The function ``analyse`` was added as an alias to ``analyze``.

Getting started with sci-analysis
---------------------------------

The documentation on how to install and use sci-analysis can be found here:

http://sci-analysis.readthedocs.io/en/latest/

Requirements
------------

* Packages: pandas, numpy, scipy, matplotlib, six
* Supports python 2.7, 3.5 and 3.6

Bugs can be reported here:

https://github.com/cmmorrow/sci-analysis/issues

