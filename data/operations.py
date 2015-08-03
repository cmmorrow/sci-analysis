import numpy as np
import vector
import data


def drop_nan(v):
    """Removes NaN values from the given sequence"""
    return v.data[~np.isnan(v.data)]


def drop_nan_intersect(first, second):
    """Creates a tuple of sequences where only non-NaN values are given"""
    c = np.logical_and(~np.isnan(first.data), ~np.isnan(second.data))
    return first.data[c], second.data[c]


def cat(list_of_data):
    """Concatenates sequences together into a Vector object"""
    output = []
    for d in list_of_data:
        if is_vector(d):
            output.append(d.data)
        else:
            output.append(d)
    return vector.Vector(output).data


def is_vector(d):
    """Checks if the argument is a Vector object"""
    if isinstance(d, vector.Vector):
        return True
    else:
        return False


def is_data(d):
    """Checks if the argument is a Data object"""
    if isinstance(d, data.Data):
        return True
    else:
        return False


def is_tuple(d):
    """Checks if the argument is a tuple"""
    if isinstance(d, tuple):
        return True
    else:
        return False


def is_iterable(d):
    """Checks if the argument is sequence-like but not a string"""
    try:
        if len(d) > 0:
            if isinstance(d, basestring):
                return False
            else:
                return True
        else:
            return False
    except TypeError:
        return False


def is_array(d):
    """ Tests if data is a numPy Array object
    """
    try:
        d.shape
        return True
    except AttributeError:
        return False


def is_dict(d):
    """ Test if data is a dictionary object
    """
    try:
        d.keys()
        return True
    except AttributeError:
        return False
