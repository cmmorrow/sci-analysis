import numpy as np
import vector


def drop_nan(v):
    return v.data[~np.isnan(v.data)]


def drop_nan_intersect(first, second):
    c = np.logical_and(~np.isnan(first.data), ~np.isnan(second.data))
    return first.data[c], second.data[c]


def cat(list_of_data):
    output = []
    for d in list_of_data:
        if is_vector(d):
            output.append(d.data)
        else:
            output.append(d)
    return vector.Vector(output).data


def is_vector(d):
    if isinstance(d, vector.Vector):
        return True
    else:
        return False


def is_data(d):
    if isinstance(d, vector.Data):
        return True
    else:
        return False


def is_tuple(d):
    if isinstance(d, tuple):
        return True
    else:
        return False


def is_iterable(d):
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
