class MinimumSizeError(Exception):
    """Thrown when the length of the Data object is less than the Graph object's _min_size property."""
    pass


class NoDataError(Exception):
    """Thrown when the Data object passed to a Graph object is empty or has no graph-able data."""
    pass

