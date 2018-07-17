"""sci_analysis module: data_operations
Functions:
    to_float - tries to convert a variable to a float.
    flatten - recursively reduces the number of dimensions to 1.
    drop_nan - removes values that are not a number from a Vector.
    drop_nan_intersect - returns only numeric values from two Vectors.
    is_vector - checks if a given sequence is a sci_analysis Vector object.
    is_data - checks if a given sequence is a sci_analysis Data object.
    is_tuple - checks if a given sequence is a tuple.
    is_iterable - checks if a given variable is iterable, but not a string.
    is_array - checks if a given sequence is a numpy Array object.
    is_dict - checks if a given sequence is a dictionary.
    is_group - checks if a given variable is a list of iterable objects.
    is_group_dict - checks if a given variable is a dictionary of iterable objects.
"""
# from __future__ import absolute_import
import six
import numpy as np
import pandas as pd


def to_float(seq):
    """
    Takes an arguement seq, tries to convert each value to a float and returns the result. If a value cannot be
    converted to a float, it is replaced by 'nan'.

    Parameters
    ----------
    seq : array-like
        The input object.

    Returns
    -------
    subseq : array_like
        seq with values converted to a float or "nan".

    >>> to_float(['1', '2', '3', 'four', '5'])
    [1.0, 2.0, 3.0, nan, 5.0]
    """
    float_list = list()
    for i in range(len(seq)):
        try:
            float_list.append(float(seq[i]))
        except ValueError:
            float_list.append(float("nan"))
        except TypeError:
            float_list.append(to_float(seq[i]))
    return float_list


def flatten(seq):
    """
    Recursively reduces the dimension of seq to one.

    Parameters
    ----------
    seq : array-like
        The input object.

    Returns
    -------
    subseq : array_like
        A flattened copy of the input object.

    Flatten a two-dimensional list into a one-dimensional list

    >>> flatten([[1, 2, 3], [4, 5, 6]])
    array([1, 2, 3, 4, 5, 6])

    Flatten a three-dimensional list into a one-dimensional list

    >>> flatten([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

    >>> flatten(([1, 2, 3], [4, 5, 6]))
    array([1, 2, 3, 4, 5, 6])

    >>> flatten(list(zip([1, 2, 3], [4, 5, 6])))
    array([1, 4, 2, 5, 3, 6])

    >>> flatten([(1, 2), (3, 4), (5, 6), (7, 8)])
    array([1, 2, 3, 4, 5, 6, 7, 8])
    """
    return np.array(seq).flatten()


def is_tuple(obj):
    """
    Checks if a given sequence is a tuple.

    Parameters
    ----------
    obj : object
        The input array.

    Returns
    -------
    test result : bool
        The test result of whether seq is a tuple or not.

    >>> is_tuple(('a', 'b'))
    True

    >>> is_tuple(['a', 'b'])
    False

    >>> is_tuple(4)
    False
    """
    return True if isinstance(obj, tuple) else False


def is_iterable(obj):
    """
    Checks if a given variable is iterable, but not a string.

    Parameters
    ----------
    obj : Any
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether variable is iterable or not.

    >>> is_iterable([1, 2, 3])
    True

    >>> is_iterable((1, 2, 3))
    True

    >>> is_iterable({'one': 1, 'two': 2, 'three': 3})
    True

    Strings arguments return False.

    >>> is_iterable('foobar')
    False

    Scalars return False.

    >>> is_iterable(42)
    False

    """
    if isinstance(obj, six.string_types):
        return False
    try:
        obj.__iter__()
        return True
    except (AttributeError, TypeError):
        return False


def is_array(obj):
    """
    Checks if a given sequence is a numpy Array object.

    Parameters
    ----------
    obj : object
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a numpy Array or not.

    >>> import numpy as np

    >>> is_array([1, 2, 3, 4, 5])
    False

    >>> is_array(np.array([1, 2, 3, 4, 5]))
    True
    """
    return hasattr(obj, 'dtype')


def is_series(obj):
    """
    Checks if a given sequence is a Pandas Series object.

    Parameters
    ----------
    obj : object
        The input argument.

    Returns
    -------
    bool

    >>> is_series([1, 2, 3])
    False

    >>> is_series(pd.Series([1, 2, 3]))
    True
    """
    return isinstance(obj, pd.Series)


def is_dict(obj):
    """
    Checks if a given sequence is a dictionary.

    Parameters
    ----------
    obj : object
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a dictionary or not.

    >>> is_dict([1, 2, 3])
    False

    >>> is_dict((1, 2, 3))
    False

    >>> is_dict({'one': 1, 'two': 2, 'three': 3})
    True

    >>> is_dict('foobar')
    False
    """
    return isinstance(obj, dict)


def is_group(seq):
    """
    Checks if a given variable is a list of iterable objects.

    Parameters
    ----------
    seq : array_like
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a list of array_like values or not.

    >>> is_group([[1, 2, 3], [4, 5, 6]])
    True

    >>> is_group({'one': 1, 'two': 2, 'three': 3})
    False

    >>> is_group(([1, 2, 3], [4, 5, 6]))
    True

    >>> is_group([1, 2, 3, 4, 5, 6])
    False

    >>> is_group({'foo': [1, 2, 3], 'bar': [4, 5, 6]})
    False
    """
    try:
        if any(is_iterable(x) for x in seq):
            return True
        else:
            return False
    except TypeError:
        return False


def is_dict_group(seq):
    """
    Checks if a given variable is a dictionary of iterable objects.

    Parameters
    ----------
    seq : array-like
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a dictionary of array_like values or not.

    >>> is_dict_group([[1, 2, 3], [4, 5, 6]])
    False

    >>> is_dict_group(([1, 2, 3], [4, 5, 6]))
    False

    >>> is_dict_group([1, 2, 3, 4, 5, 6])
    False

    >>> is_dict_group({'foo': [1, 2, 3], 'bar': [4, 5, 6]})
    True
    """
    try:
        if is_group(list(seq.values())):
            return True
        else:
            return False
    except (AttributeError, TypeError):
        return False


def is_number(obj):
    """
    Checks if the given object is a number.

    Parameters
    ----------
    obj : Any
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether obj can be converted to a number or not.

    >>> is_number(3)
    True

    >>> is_number(1.34)
    True

    >>> is_number('3')
    True

    >>> is_number(np.array(3))
    True

    >>> is_number('a')
    False

    >>> is_number([1, 2, 3])
    False

    >>> is_number(None)
    False
    """
    try:
        float(obj)
        return True
    except (ValueError, TypeError):
        return False
