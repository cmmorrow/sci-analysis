============
sci-analysis
============

An easy to use and powerful python-based data exploration and analysis tool

---------------
Current Version
---------------

2.1 --- Released March 30, 2018

.. image:: https://img.shields.io/pypi/v/sci_analysis.svg
   :target: https://pypi.python.org/pypi/sci_analysis
.. image:: https://img.shields.io/pypi/format/sci_analysis.svg
   :target: https://pypi.python.org/pypi/sci_analysis
.. image:: https://img.shields.io/pypi/pyversions/sci_analysis.svg
   :target: https://pypi.python.org/pypi/sci_analysis
.. image:: https://travis-ci.org/cmmorrow/sci-analysis.svg?branch=2.1.0
   :target: https://travis-ci.org/cmmorrow/sci-analysis
.. image:: https://coveralls.io/repos/github/cmmorrow/sci-analysis/badge.svg?branch=2.1.0
   :target: https://coveralls.io/github/cmmorrow/sci-analysis?branch=2.0.0

What is sci-analysis?
---------------------

sci-analysis is a python package for quickly performing statistical data analysis. It provides a graphical representation of the supplied data as well as the statistical analysis. sci-analysis is smart enough to determine the correct analysis and tests to perform based on the shape of the data you provide, as well as how the data is distributed.

The types of analysis that can be performed are histograms of numeric or categorical data, bivariate analysis of two numeric data vectors, and one-way analysis of variance.

What's new in sci-analysis version 2.0?
---------------------------------------

* Version 2.1 makes improvements to Statistical output and plots.
* Tukey-Kramer circles were added to the Oneway analysis plot.
* Grand Mean and Grand Median were added to the Oneway analysis plot.
* Overall Statistics were added to Oneway analysis.
* Overall Statistics were added to Categorical analysis.
* The Categorical analysis graph was changed to improve the appearance.

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

