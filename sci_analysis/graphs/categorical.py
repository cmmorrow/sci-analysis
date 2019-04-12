import math

# matplotlib imports
from matplotlib.pyplot import show, xticks, savefig, close, subplots, subplots_adjust

# local imports
from .base import Graph
from ..data import Categorical, is_group, is_categorical
from ..analysis.exc import MinimumSizeError, NoDataError


class CategoricalGraph(Graph):

    def __init__(self, *args, **kwargs):
        order = kwargs['order'] if 'order' in kwargs else None
        dropna = kwargs['dropna'] if 'dropna' in kwargs else False
        seq_name = kwargs['name'] if 'name' in kwargs else None
        data = list()
        for d in args:
            if not d:
                raise NoDataError('Cannot draw graph because there is no data.')
            new = d if is_categorical(d) else Categorical(d, name=seq_name, order=order, dropna=dropna)
            if len(new) <= self._min_size:
                raise MinimumSizeError('Length of data is less than the minimum size {}.'.format(self._min_size))
            data.append(new)
        if not is_group(data):
            raise NoDataError('Cannot draw graph because there is no data.')
        if len(data) == 1:
            data = data[0]
        super(CategoricalGraph, self).__init__(data, **kwargs)
        self.draw()

    def draw(self):
        """
        Prepares and displays the graph based on the set class members.
        """
        raise NotImplementedError


class GraphFrequency(CategoricalGraph):

    _xsize = 7
    _ysize = 5.5

    def __init__(self, data, **kwargs):

        self._percent = kwargs['percent'] if 'percent' in kwargs else False
        self._vertical = kwargs['vertical'] if 'vertical' in kwargs else True
        self._grid = kwargs['grid'] if 'grid' in kwargs else False
        self._labels = kwargs['labels'] if 'labels' in kwargs else True
        self._title = kwargs['title'] if 'title' in kwargs else 'Frequencies'
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        order = kwargs['order'] if 'order' in kwargs else None
        dropna = kwargs['dropna'] if 'dropna' in kwargs else False
        yname = 'Percent' if self._percent else 'Frequency'
        name = 'Categories'
        if 'name' in kwargs:
            name = kwargs['name']
        elif 'xname' in kwargs:
            name = kwargs['xname']

        super(GraphFrequency, self).__init__(data, xname=name, yname=yname, order=order, dropna=dropna)

    def add_numeric_labels(self, bars, axis):
        if self._vertical:
            if len(bars) < 3:
                size = 'xx-large'
            elif len(bars) < 9:
                size = 'x-large'
            elif len(bars) < 21:
                size = 'large'
            elif len(bars) < 31:
                size = 'medium'
            else:
                size = 'small'
            for bar in bars:
                x_pos = bar.get_width()
                y_pos = bar.get_y() + bar.get_height() / 2.
                x_off = x_pos + 0.05
                adjust = .885 if self._percent else .95
                if not self._percent and x_pos != 0:
                    adjust = adjust - math.floor(math.log10(x_pos)) * .035
                label = '{:.1f}'.format(x_pos) if self._percent else '{}'.format(x_pos)
                col = 'k'
                if x_pos != 0 and (x_off / axis.get_xlim()[1]) > .965 - math.floor(math.log10(x_pos)) * .02:
                    x_off = x_pos * adjust
                    col = 'w'
                axis.annotate(label,
                              xy=(x_pos, y_pos),
                              xytext=(x_off, y_pos),
                              va='center',
                              color=col,
                              size=size)
        else:
            if len(bars) < 21:
                size = 'medium'
            elif len(bars) < 31:
                size = 'small'
            else:
                size = 'x-small'
            for bar in bars:
                y_pos = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2.
                y_off = y_pos + 0.05
                label = '{:.1f}'.format(y_pos) if self._percent else '{}'.format(y_pos)
                col = 'k'
                if (y_off / axis.get_ylim()[1]) > 0.95:
                    y_off = y_pos * .95
                    col = 'w'
                axis.annotate(label,
                              xy=(x_pos, y_pos),
                              xytext=(x_pos, y_off),
                              ha='center',
                              size=size,
                              color=col)

    def draw(self):

        freq = self._data.percents if self._percent else self._data.counts
        categories = self._data.categories.tolist()
        nbars = tuple(range(1, len(freq) + 1))
        grid_props = dict(linestyle='-', which='major', color='grey', alpha=0.75)
        bar_props = dict(color=self.get_color(0), zorder=3)

        # Create the figure and axes
        if self._vertical:
            f, ax = subplots(figsize=(self._ysize, self._xsize))
        else:
            f, ax = subplots(figsize=(self._xsize, self._ysize))

        # Set the title
        f.suptitle(self._title, fontsize=14)

        # Create the graph, grid and labels
        if self._grid:
            ax.xaxis.grid(True, **grid_props) if self._vertical else ax.yaxis.grid(True, **grid_props)
        categories = ['{}...'.format(cat[:18]) if len(str(cat)) > 20 else cat for cat in categories]
        max_len = max([len(str(cat)) for cat in categories])
        offset = max_len / 5 * 0.09
        if self._vertical:
            bars = ax.barh(nbars, freq.tolist(), **bar_props)
            ax.set_xlabel(self._yname)
            ax.set_yticks(nbars)
            ax.set_yticklabels(categories)
            subplots_adjust(left=offset)
            ax.invert_yaxis()
        else:
            bars = ax.bar(nbars, freq.tolist(), **bar_props)
            ax.set_ylabel(self._yname)
            ax.set_xticks(nbars)
            angle = 90 if len(nbars) > 15 else 60
            xticks(rotation=angle)
            ax.set_xticklabels(categories)
            subplots_adjust(bottom=offset)
        if self._labels:
            self.add_numeric_labels(bars, ax)

        # Save the figure to disk or display
        if self._save_to:
            savefig(self._save_to)
            close(f)
        else:
            show()
        pass
