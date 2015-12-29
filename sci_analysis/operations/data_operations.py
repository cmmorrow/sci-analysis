"""sci_analysis module: data_operations
Functions:
    to_float - tries to convert a variable to a float.
    flatten - reduces the number of dimensions by 1.
    clean - [depricated] Alias for drop_nan or drop_nan_intersect.
    strip - [depricated] Converts an array-like object to a Vector.
    drop_nan - removes values that are not a number from a Vector.
    drop_nan_intersect - returns only numeric values from two Vectors.
    is_vector - checks if an array-like object is a Vector object.
    is_data - checks if an array-like object is a Data object.
    is_tuple - checks if a sequence is a tuple.
    is_iterable - checks if a variable is iterable.
    is_array - checks if an array-like object is a numPy array.
    is_dict - checks if an array-like object is a dict.
    is_group - checks if a variable is a list of iterables.
    is_group_dict - checks if a variable is a dict of iterables.
"""
from __future__ import absolute_import

import numpy as np

from ..data.data import Data
try:
    from ..data.vector import Vector
except ImportError:
    import sys
    Vector = sys.modules["sci_analysis.data.vector.Vector"]


def to_float(d):
    """Takes a data argument d, tries to convert d to a float and returns
    the result. Otherwise, "nan" is returned.

    :param d: Data of unspecified type
    :return: d converted to a float or "nan"
    """
    float_list = []
    for i in range(len(d)):
        try:
            float_list.append(float(d[i]))
        except (ValueError, TypeError):
            float_list.append(float("nan"))
    return float_list


def flatten(d):
    """Reduces the dimension of data d by one.

    :param d: A sequence of data
    :return: The flattened sequence
    """
    flat = []
    for row in d:
        if is_iterable(row):
            for col in row:
                flat.append(col)
        else:
            flat.append(row)
    return flat


def clean(x, y=list()):
    """This is a deprecated function from sci_analysis 1.2 which now just converts
    the args to a Vector and passes it through drop_nan.

    :param x: A sequence like object
    :param y: A sequence like object
    :return: x and y as a Vector object
    """
    if len(y) > 0:
        return drop_nan_intersect(Vector(x), Vector(y))
    else:
        return drop_nan(Vector(x))


def strip(d):
    """This is a deprecated function from sci_analysis 1.2 which now just converts
    d to a Vector.

    :param d: A sequence like object
    :return: d as a numPy Array
    """
    return Vector(d).data


def drop_nan(v):
    """ Removes NaN values from the given sequence v and returns the equivalent
    Vector object. The length of the returned Vector is the length of v minus
    the number of "nan" values removed from v.

    :param v: A sequence like object
    :return: A vector representation of v with "nan" values removed
    """
    return Vector(v.data[~np.isnan(v.data)])


def drop_nan_intersect(first, second):
    """Takes two sequence like arguments first and second, and creates a
    tuple of sequences where only non-NaN values are given. This is accomplished
    by removing values that are "nan" on matching indicies of either first or second.

    :param first: A sequence like object
    :param second: A sequence like object
    :return: A two element tuple of Vector objects with "nan" values removed
    """
    c = np.logical_and(~np.isnan(first.data), ~np.isnan(second.data))
    return Vector(first.data[c]), Vector(second.data[c])


# def cat(list_of_data):
#    """Concatenates sequences together into a Vector object"""
#    output = []
#    for d in list_of_data:
#        if is_vector(d):
#            output.append(d.data)
#        else:
#            output.append(d)
#    return vector.Vector(output).data


def is_vector(d):
    """Checks if the argument is a Vector object.

    :param d: A variable of unknown type
    :return: True or False
    """
    if isinstance(d, Vector):
        return True
    else:
        return False


def is_data(d):
    """Checks if the argument is a Data object.

    :param d: A variable of unknown type
    :return: True or False
    """
    if isinstance(d, Data):
        return True
    else:
        return False


def is_tuple(d):
    """Checks if the argument is a tuple.

    :param d: A variable of unknown type
    :return: True or False
    """
    if isinstance(d, tuple):
        return True
    else:
        return False


def is_iterable(d):
    """Checks if the argument is sequence-like but not a string.

    :param d: A variable of unknown type
    :return: True or False
    """
    try:
        d.__iter__()
        return True
    except (AttributeError, TypeError):
        return False


def is_array(d):
    """Tests if the argument is a numPy Array object.

    :param d: A variable of unknown type
    :return: True or False
    """
    try:
        d.dtype
        return True
    except AttributeError:
        return False


def is_dict(d):
    """Tests if the argument is a dictionary object.

    :param d: A variable of unknown type
    :return: True or False
    """
    """"""
    try:
        list(d.items())
        return True
    except AttributeError:
        return False


def is_group(d):
    """Tests if the argument is a list of iterables.

    :param d: A variable of unknown type
    :return: True or False
    """
    try:
        if any(is_iterable(x) for x in d):
            return True
        else:
            return False
    except TypeError:
        return False


def is_dict_group(d):
    """Tests if the argument is a dict of iterables

    :param d: A variable of unknown type
    :return: True or False
    """
    try:
        if is_group(list(d.values())):
            return True
        else:
            return False
    except (AttributeError, TypeError):
        return False
