
sci-analysis
============

An easy to use and powerful python-based data exploration and analysis tool

Current Version:
----------------

2.2 --- Released January 5, 2019


.. image:: https://img.shields.io/pypi/v/sci_analysis.svg
   :target: https://pypi.python.org/pypi/sci_analysis
   :alt: PyPI


.. image:: https://img.shields.io/pypi/format/sci_analysis.svg
   :target: https://pypi.python.org/pypi/sci_analysis
   :alt: PyPI


.. image:: https://img.shields.io/pypi/pyversions/sci_analysis.svg
   :target: https://pypi.python.org/pypi/sci_analysis
   :alt: PyPI


.. image:: https://travis-ci.org/cmmorrow/sci-analysis.svg?branch=master
   :target: https://travis-ci.org/cmmorrow/sci-analysis
   :alt: Build Status


.. image:: https://coveralls.io/repos/github/cmmorrow/sci-analysis/badge.svg?branch=master
   :target: https://coveralls.io/github/cmmorrow/sci-analysis?branch=master
   :alt: Coverage Status


What is sci-analysis?
---------------------

sci-analysis is a python package for quickly performing exploratory data analysis (EDA). It aims to make performing EDA easier for newcomers and experienced data analysts alike by abstracting away the specific SciPy, NumPy, and Matplotlib commands. This is accomplished by using sci-analysis's ``analyze()`` function.

The main features of sci-analysis are:


* Fast EDA with the analyze() function.
* Great looking graphs without writing several lines of matplotlib code.
* Automatic use of the most appropriate hypothesis test for the supplied data.
* Automatic handling of missing values.

Currently, sci-analysis is capable of performing four common statistical analysis techniques:


* Histograms and summary of numeric data
* Histograms and frequency of categorical data
* Bivariate and linear regression analysis
* Location testing

What's new in sci-analysis version 2.2?
---------------------------------------


* Version 2.2 adds the ability to add data labels to scatter plots.
* The default behavior of the histogram and statistics was changed from assuming a sample, to assuming a population.
* Fixed a bug involving the Mann Whitney U test, where the minimum size was set incorrectly.
* Verified compatibility with python 3.7.

Requirements
------------


* Packages: pandas, numpy, scipy, matplotlib, six
* Supports python 2.7, 3.5, 3.6, and 3.7

Bugs can be reported here:

`https://github.com/cmmorrow/sci-analysis/issues <https://github.com/cmmorrow/sci-analysis/issues>`_

Documentation
-------------

The documentation on how to install and use sci-analysis can be found here:

`http://sci-analysis.readthedocs.io/en/latest/ <http://sci-analysis.readthedocs.io/en/latest/>`_

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

    Getting Started With sci-analysis <getting_started>
    Using sci-analysis <using_sci_analysis>
    Using sci-analysis With Pandas <pandas>
    Analysis Types <analysis_types>
