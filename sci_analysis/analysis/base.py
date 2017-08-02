"""Module: base.py
Classes:
    Analysis - Generic analysis root class.
    Test - Generic statistical test class.
    GroupTest - Perform a test on multiple vectors that are passed as a tuple of arbitrary length.
    Comparison - Perform a test on two independent vectors of equal length.
    NormTest - Tests for whether data is normally distributed or not.
    GroupNormTest - Tests a group of data to see if they are normally distributed or not.
    TTest - Performs a T-Test on the two provided vectors.
    LinearRegression - Performs a linear regression between two vectors.
    Correlation - Performs a pearson or spearman correlation between two vectors.
    Anova - Performs a one-way ANOVA on a group of vectors.
    Kruskal - Performs a non-parametric Kruskal-Wallis test on a group of vectors.
    EqualVariance - Checks a group of vectors for equal variance.
    VectorStatistics - Reports basic summary stats for a provided vector.
    GroupStatistics - Reports basic summary stats for a group of vectors.
Functions:
    analyze - Magic method for performing quick data analysis.
"""
# Python3 compatability
from __future__ import absolute_import
from __future__ import print_function

# Local imports
from analysis.func import std_output


class Analysis(object):
    """Generic analysis root class.

    Members:
        _data - the data used for analysis.
        _display - flag for whether to display the analysis output.
        _results - A dict of the results of the test.

    Methods:
        logic - This method needs to run the analysis, set the results member, and display the output at bare minimum.
        run - This method should return the results of the specific analysis.
        output - This method shouldn't return a value and only produce a side-effect.
    """

    _name = "Analysis"

    def __init__(self, data, display=True):
        """Initialize the data and results members.

        Override this method to initialize additional members or perform
        checks on data.
        """
        self._data = data
        self._display = display
        self._results = {}

    @property
    def name(self):
        """The name of the test class"""
        return self._name

    @property
    def data(self):
        """The data used for analysis"""
        return self._data

    @property
    def results(self):
        """A dict of the results returned by the run method"""
        return self._results

    def logic(self):
        """This method needs to run the analysis, set the results member, and
        display the output at bare minimum.

        Override this method to modify the execution sequence of the analysis.
        """
        if self._data is None:
            return
        self.run()
        if self._display:
            print(self)

    def run(self):
        """This method should perform the specific analysis and set the results dict.

        Override this method to perform a specific analysis or calculation.
        """
        raise NotImplementedError

    def __str__(self):
        return std_output(self._name, self._results, self._results.keys())
