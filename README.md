# sci-analysis
An easy to use and powerful python-based data exploration and analysis tool

## Current Version:
2.2 --- Released January 5, 2019

[![PyPI](https://img.shields.io/pypi/v/sci_analysis.svg)](https://pypi.python.org/pypi/sci_analysis)
[![PyPI](https://img.shields.io/pypi/format/sci_analysis.svg)](https://pypi.python.org/pypi/sci_analysis)
[![PyPI](https://img.shields.io/pypi/pyversions/sci_analysis.svg)](https://pypi.python.org/pypi/sci_analysis)
[![Build Status](https://travis-ci.org/cmmorrow/sci-analysis.svg?branch=master)](https://travis-ci.org/cmmorrow/sci-analysis)
[![Coverage Status](https://coveralls.io/repos/github/cmmorrow/sci-analysis/badge.svg?branch=master)](https://coveralls.io/github/cmmorrow/sci-analysis?branch=master)

### What is sci-analysis?
sci-analysis is a python package for quickly performing statistical data analysis. It provides a graphical representation of the supplied data as well as the statistical analysis. sci-analysis is smart enough to determine the correct analysis and tests to perform based on the shape of the data you provide, as well as how the data is distributed.

The types of analysis that can be performed are histograms of numeric or categorical data, bi-variate analysis of two numeric data vectors, and one-way analysis of variance.

### What's new in sci-analysis version 2.2?

* Version 2.2 adds the ability to add data labels to scatter plots.
* The default behavior of the histogram and statistics was changed from assuming a sample, to assuming a population.
* Fixed a bug involving the Mann Whitney U test, where the minimum size was set incorrectly.
* Verified compatibility with python 3.7.

### Getting started with sci-analysis
The documentation on how to install and use sci-analysis can be found here:

[http://sci-analysis.readthedocs.io/en/latest/](http://sci-analysis.readthedocs.io/en/latest/)


### Requirements
* Packages: pandas, numpy, scipy, matplotlib, six
* Supports python 2.7, 3.5, 3.6, and 3.7

Bugs can be reported here:

[https://github.com/cmmorrow/sci-analysis/issues](https://github.com/cmmorrow/sci-analysis/issues)

