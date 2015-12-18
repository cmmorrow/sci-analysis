"""This is the sci_analysis package.
Sub packages:
    data - sci_analysis data types and functions
    analysis - sci_analysis test and calculation classes and functions
    graphs - graphing classes

The data, analysis and graphs modules are loaded.
The following classes are imported so that they are exposed at a high level: Vector,
NormTest, GroupNormTest, LinearRegression, Correlation, Anova, Kruskal, EqualVariance,
VectorStatistics, GroupStatistics, GraphHisto, GraphScatter and GraphBoxplot. The
following methods are imported so that they are exposed at a high level: analyze,
clean and strip.
"""
from data.vector import Vector
from data.operations import clean, strip
from analysis import analyze, NormTest, GroupNormTest, LinearRegression, Correlation,\
    Anova, Kruskal, EqualVariance, VectorStatistics, GroupStatistics
from graphs import GraphHisto, GraphScatter, GraphBoxplot

__all__ = ["data", "analysis", "graphs"]
