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
import six

# from ..graphs.graph import Graph


def to_float(seq):
    """Takes a data argument d, tries to convert d to a float and returns
    the result. Otherwise, "nan" is returned.

    :param seq: Data of unspecified type
    :return: d converted to a float or "nan"
    """
    float_list = []
    for i in range(len(seq)):
        try:
            float_list.append(float(seq[i]))
        except ValueError:
            float_list.append(float("nan"))
        except TypeError:
            float_list.append(to_float(seq[i]))
    return float_list


def flatten(seq):
    """Reduces the dimension of data d by one.

    :param seq: A sequence of data
    :return: The flattened sequence
    """
    flat = []
    for row in seq:
        if is_iterable(row):
            for col in flatten(row):
                flat.append(col)
        else:
            flat.append(row)
    return flat


# def is_graph(obj):
#     """Checks if the argument is a Graph object
#
#     :param obj: A variable of unknown type
#     :return: True or False
#     """
#     return True if isinstance(obj, Graph) else False


def is_tuple(seq):
    """Checks if the argument is a tuple.

    :param seq: A variable of unknown type
    :return: True or False
    """
    if isinstance(seq, tuple):
        return True
    else:
        return False


def is_iterable(seq):
    """Checks if the argument is sequence-like but not a string.

    :param seq: A variable of unknown type
    :return: True or False
    """
    if isinstance(seq, six.string_types):
        return False
    try:
        seq.__iter__()
        return True
    except (AttributeError, TypeError):
        return False


def is_array(seq):
    """Tests if the argument is a numPy Array object.

    :param seq: A variable of unknown type
    :return: True or False
    """
    return hasattr(seq, 'dtype')


def is_dict(seq):
    """Tests if the argument is a dictionary object.

    :param seq: A variable of unknown type
    :return: True or False
    """
    try:
        seq.values()
        return True
    except (AttributeError, TypeError):
        return False


def is_group(seq):
    """Tests if the argument is a list of iterables.

    :param seq: A variable of unknown type
    :return: True or False
    """
    try:
        if any(is_iterable(x) for x in seq):
            return True
        else:
            return False
    except TypeError:
        return False


def is_dict_group(seq):
    """Tests if the argument is a dict of iterables

    :param seq: A variable of unknown type
    :return: True or False
    """
    try:
        if is_group(list(seq.values())):
            return True
        else:
            return False
    except (AttributeError, TypeError):
        return False
