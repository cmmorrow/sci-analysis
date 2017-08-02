# Numpy imports
from numpy import float_, int_


class MinimumSizeError(Exception):
    """Thrown when the length of the Data object is less than the Graph object's _min_size property."""
    pass


class NoDataError(Exception):
    """Thrown when the Data object passed to a Graph object is empty or has no graph-able data."""
    pass


def std_output(name, results, order, precision=4, spacing=14):
    """

    Parameters
    ----------
    name : str
        The name of the analysis report.
    results : dict or list
        The input dict or list to print.
    order : list
        The list of keys in results to display and the order to display them in.
    precision : int
        The number of decimal places to show for float values.
    spacing : int
        The max number of characters for each printed column.

    Returns
    -------
    output_string : str
        The report to be printed to stdout.
    """

    def format_header(col_names):
        line = ""
        for n in col_names:
            line += '{:{}s}'.format(n, spacing)
        return line

    def format_row(_row, _order):
        line = ""
        for column in _order:
            value = _row[column]
            t = type(value)
            if t in [float, float_]:
                line += '{:< {}.{}f}'.format(value, spacing, precision)
            elif t in [float, float_]:
                line += '{:< {}d}'.format(value, spacing)
            else:
                line += '{:<{}s}'.format(str(value), spacing)
        return line

    def format_items(label, value):
        if type(value) in {float, float_}:
            line = '{:{}s}'.format(label, max_length) + ' = ' + '{:< .{}f}'.format(value, precision)
        elif type(value) in {int, int_}:
            line = '{:{}s}'.format(label, max_length) + ' = ' + '{:< d}'.format(value)
        else:
            line = '{:{}s}'.format(label, max_length) + ' = ' + str(value)
        return line

    table = list()
    header = ''

    if isinstance(results, list):
        header = format_header(order)
        for row in results:
            table.append(format_row(row, order))
    elif isinstance(results, dict):
        max_length = max([len(label) for label in results.keys()])
        for key in order:
            table.append(format_items(key, results[key]))

    out = [
        '',
        '',
        name,
        '-' * len(name),
        ''
    ]
    if len(header) > 0:
        out.extend([
            header,
            '-' * len(header)
        ])
    out.append('\n'.join(table))
    return '\n'.join(out)
