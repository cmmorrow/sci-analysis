from __future__ import absolute_import
import numpy as np
from . import vector
from . import data


def to_float(d):
        """Converts values in data to float and returns a copy"""
        float_list = []
        for i in range(len(d)):
            try:
                float_list.append(float(d[i]))
            except (ValueError, TypeError):
                float_list.append(float("nan"))
        return float_list


def flatten(d):
        """Reduce the dimension of data by one"""
        flat = []
        for row in d:
            if is_iterable(row):
                for col in row:
                    flat.append(col)
            else:
                flat.append(row)
        return flat


def clean(x, y=list()):
    """This is a deprecated function from 1.2 which now just converts the args
    to a Vector and passes it through drop_nan"""
    if len(y) > 0:
        return drop_nan_intersect(vector.Vector(x), vector.Vector(y))
    else:
        return drop_nan(vector.Vector(x))


def strip(d):
    """This is a deprecated function from 1.2 which now just converts d to a Vector"""
    return vector.Vector(d).data


def drop_nan(v):
    """Removes NaN values from the given sequence"""
    return vector.Vector(v.data[~np.isnan(v.data)])


def drop_nan_intersect(first, second):
    """Creates a tuple of sequences where only non-NaN values are given"""
    c = np.logical_and(~np.isnan(first.data), ~np.isnan(second.data))
    return vector.Vector(first.data[c]), vector.Vector(second.data[c])


#def cat(list_of_data):
#    """Concatenates sequences together into a Vector object"""
#    output = []
#    for d in list_of_data:
#        if is_vector(d):
#            output.append(d.data)
#        else:
#            output.append(d)
#    return vector.Vector(output).data


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
        d.__iter__()
        return True
    except (AttributeError, TypeError):
        return False


def is_array(d):
    """Tests if data is a numPy Array object"""
    try:
        d.dtype
        return True
    except AttributeError:
        return False


def is_dict(d):
    """Test if data is a dictionary object"""
    try:
        list(d.items())
        return True
    except AttributeError:
        return False


def is_group(d):
    """Test if data is a list of iterables"""
    try:
        if any(is_iterable(x) for x in d):
            return True
        else:
            return False
    except TypeError:
        return False


def is_dict_group(d):
    """Test if data is a dict of iterables"""
    try:
        if is_group(list(d.values())):
            return True
        else:
            return False
    except (AttributeError, TypeError):
        return False
