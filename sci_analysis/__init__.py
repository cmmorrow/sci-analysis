"""This is the sci_analysis package.
Sub packages:
    data - sci_analysis data types
    analysis - sci_analysis test and calculation classes and functions
    operations - sci_analysis general functions
    graphs - graphing classes

The analysis and operations modules are loaded.
The following classes are imported so that they are exposed at a high level: Vector,
NormTest, GroupNormTest, LinearRegression, Correlation, Anova, Kruskal, EqualVariance,
VectorStatistics, GroupStatistics, GraphHisto, GraphScatter and GraphBoxplot. The
following methods are imported so that they are exposed at a high level: analyze,
clean and strip.
"""

from . import analysis
from .analysis import Comparison, NormTest, GroupNormTest, TTest, LinearRegression,\
    Correlation, Anova, Kruskal, EqualVariance, VectorStatistics, GroupStatistics,\
    analyze, Vector
from .operations import clean, strip

__all__ = ["data", "analysis", "graphs"]
