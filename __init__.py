__all__ = ["data", "analysis", "graphs"]
from data.vector import Vector
from analysis import analyze, NormTest, GroupNormTest, LinearRegression, Correlation, Anova, Kruskal, EqualVariance, VectorStatistics, GroupStatistics
from graphs import GraphHisto, GraphScatter, GraphBoxplot