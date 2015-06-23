from . import vector
import numpy as np


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
