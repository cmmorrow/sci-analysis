"""sci_analysis package: data
Modules:
    data - the sci_analysis Data class
    operations - sci_analysis Data functions
    vector - the sci_analysis Vector class
"""
from .data import Data, Vector, Categorical, is_data, is_numeric, is_categorical, is_vector, \
    EmptyVectorError, UnequalVectorLengthError, NumberOfCategoriesWarning
