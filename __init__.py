from data.vector import Vector
from data.operations import clean, strip
from analysis import analyze, NormTest, GroupNormTest, LinearRegression, Correlation,\
    Anova, Kruskal, EqualVariance, VectorStatistics, GroupStatistics
from graphs import GraphHisto, GraphScatter, GraphBoxplot

__all__ = ["data", "analysis", "graphs"]
