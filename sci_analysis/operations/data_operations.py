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
from __future__ import absolute_import
import six


def to_float(seq):
    """
    Takes an arguement seq, tries to convert each value to a float and returns the result. If a value cannot be
    converted to a float, it is replaced by 'nan'.

    Parameters
    ----------
    seq : array_like
        The input object.

    Returns
    -------
    subseq : array_like
        seq with values converted to a float or "nan".

    Examples
    --------
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
    seq : array_like
        The input object.

    Returns
    -------
    subseq : array_like
        A flattened copy of the input object.

    Examples
    --------

    Flatten a two-dimensional list into a one-dimensional list

    >>> flatten([[1, 2, 3], [4, 5, 6]])
    [1, 2, 3, 4, 5, 6]

    Flatten a three-dimensional list into a one-dimensional list

    >>> flatten([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    """
    flat = list()
    for row in seq:
        if is_iterable(row):
            for col in flatten(row):
                flat.append(col)
        else:
            flat.append(row)
    return flat


def is_tuple(seq):
    """
    Checks if a given sequence is a tuple.

    Parameters
    ----------
    seq : array_like
        The input array.

    Returns
    -------
    test result : bool
        The test result of whether seq is a tuple or not.

    Examples
    --------
    >>> is_tuple(('a', 'b'))
    True

    >>> is_tuple(['a', 'b'])
    False

    >>> is_tuple(4)
    False
    """
    return True if isinstance(seq, tuple) else False


def is_iterable(variable):
    """
    Checks if a given variable is iterable, but not a string.

    Parameters
    ----------
    variable : object
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether variable is iterable or not.

    Examples
    --------
    >>> is_iterable([1, 2, 3])
    True

    >>> is_iterable((1, 2, 3))
    True

    >>> is_iterable({'one': 1, 'two': 2, 'three': 3})
    True

    Strings arguments return False.

    >>> is_iterable('foobar')
    False

    """
    if isinstance(variable, six.string_types):
        return False
    try:
        variable.__iter__()
        return True
    except (AttributeError, TypeError):
        return False


def is_array(seq):
    """
    Checks if a given sequence is a numpy Array object.

    Parameters
    ----------
    seq : array_like
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a numpy Array or not.

    Examples
    --------
    >>> import numpy as np

    >>> is_array([1, 2, 3, 4, 5])
    False

    >>> is_array(np.array([1, 2, 3, 4, 5]))
    True
    """
    return hasattr(seq, 'dtype')


def is_dict(seq):
    """
    Checks if a given sequence is a dictionary.

    Parameters
    ----------
    seq : array_like
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a dictionary or not.

    Examples
    --------
    >>> is_dict([1, 2, 3])
    False

    >>> is_dict((1, 2, 3))
    False

    >>> is_dict({'one': 1, 'two': 2, 'three': 3})
    True

    >>> is_dict('foobar')
    False
    """
    try:
        seq.values()
        return True
    except (AttributeError, TypeError):
        return False


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

    Examples
    --------
    >>> is_group([[1, 2, 3], [4, 5, 6]])
    True

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
    seq : array_like
        The input argument.

    Returns
    -------
    test result : bool
        The test result of whether seq is a dictionary of array_like values or not.

    Examples
    --------
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
