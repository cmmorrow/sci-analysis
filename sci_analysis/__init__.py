"""This is the sci_analysis package.
Sub packages:
    data - sci_analysis data types
    analysis - sci_analysis test and calculation classes and functions
    graphs - graphing classes

The analysis and operations modules are loaded.
The following classes are imported so that they are exposed at a high level: Vector,
NormTest, GroupNormTest, LinearRegression, Correlation, Anova, Kruskal, EqualVariance,
VectorStatistics, GroupStatistics, GraphHisto, GraphScatter and GraphBoxplot. The
following methods are imported so that they are exposed at a high level: analyze,
clean and strip.
"""
# from __future__ import absolute_import

# from sci_analysis.analysis import analysis
# from sci_analysis.graphs import graph
# from sci_analysis.data import data
__all__ = ["data", "analysis", "graphs"]
from .analysis import analyze, analyse
# from .analysis.analysis import Comparison, NormTest, TTest, LinearRegression,\
#     Correlation, Anova, Kruskal, EqualVariance, VectorStatistics, GroupStatistics
