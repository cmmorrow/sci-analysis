"""Module: base.py
Classes:
    Analysis - Generic analysis root class.
    Test - Generic statistical test class.
    GroupTest - Perform a test on multiple vectors that are passed as a tuple of arbitrary length.
    Comparison - Perform a test on two independent vectors of equal length.
    NormTest - Tests for whether data is normally distributed or not.
    GroupNormTest - Tests a group of data to see if they are normally distributed or not.
    TTest - Performs a T-Test on the two provided vectors.
    LinearRegression - Performs a linear regression between two vectors.
    Correlation - Performs a pearson or spearman correlation between two vectors.
    Anova - Performs a one-way ANOVA on a group of vectors.
    Kruskal - Performs a non-parametric Kruskal-Wallis test on a group of vectors.
    EqualVariance - Checks a group of vectors for equal variance.
    VectorStatistics - Reports basic summary stats for a provided vector.
    GroupStatistics - Reports basic summary stats for a group of vectors.
Functions:
    analyze - Magic method for performing quick data analysis.
"""

# Numpy imports
from numpy import float_, int_


class Analysis(object):
    """Generic analysis root class.

    Members:
        _data - the data used for analysis.
        _display - flag for whether to display the analysis output.
        _results - A dict of the results of the test.

    Methods:
        logic - This method needs to run the analysis, set the results member, and display the output at bare minimum.
        run - This method should return the results of the specific analysis.
        output - This method shouldn't return a value and only produce a side-effect.
    """

    _name = "Analysis"

    def __init__(self, data, display=True):
        """Initialize the data and results members.

        Override this method to initialize additional members or perform
        checks on data.
        """
        self._data = data
        self._display = display
        self._results = {}

    @property
    def name(self):
        """The name of the test class"""
        return self._name

    @property
    def data(self):
        """The data used for analysis"""
        return self._data

    @property
    def results(self):
        """A dict of the results returned by the run method"""
        return self._results

    def logic(self):
        """This method needs to run the analysis, set the results member, and
        display the output at bare minimum.

        Override this method to modify the execution sequence of the analysis.
        """
        if self._data is None:
            return
        self.run()
        if self._display:
            print(self)

    def run(self):
        """This method should perform the specific analysis and set the results dict.

        Override this method to perform a specific analysis or calculation.
        """
        raise NotImplementedError

    def __str__(self):
        return std_output(self._name, self._results, tuple(self._results.keys()))


def std_output(name, results, order, precision=4, spacing=14):
    """

    Parameters
    ----------
    name : str
        The name of the analysis report.
    results : dict or list
        The input dict or list to print.
    order : list or tuple
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
