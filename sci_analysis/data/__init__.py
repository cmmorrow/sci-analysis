"""sci_analysis package: data
Modules:
    data - the sci_analysis Data class
    operations - sci_analysis Data functions
"""
from .data import Data, is_data

from .data_operations import (to_float, flatten, is_tuple, is_iterable, is_array, is_series, is_dict, is_group,
                              is_dict_group, is_number)

from .categorical import NumberOfCategoriesWarning, is_categorical, Categorical

from .numeric import EmptyVectorError, UnequalVectorLengthError, is_numeric, is_vector, Numeric, Vector
