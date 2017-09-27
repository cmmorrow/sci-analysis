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
            new = d if is_categorical(d) else Categorical(d, name=seq_name, order=order, dropna=dropna)
            if new.is_empty():
                raise NoDataError('Cannot draw graph because there is no data.')
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
    _ysize = 6

    def __init__(self, data, **kwargs):

        self._percent = kwargs['percent'] if 'percent' in kwargs else False
        self._vertical = kwargs['vertical'] if 'vertical' in kwargs else True
        self._title = kwargs['title'] if 'title' in kwargs else 'Frequencies'
        self._save_to = kwargs['save_to'] if 'save_to' in kwargs else None
        order = kwargs['order'] if 'order' in kwargs else None
        dropna = kwargs['dropna'] if 'dropna' in kwargs else False
        yname = 'Percent' if self._percent else 'Freq'
        name = 'Categories'
        if 'name' in kwargs:
            name = kwargs['name']
        elif 'xname' in kwargs:
            name = kwargs['xname']

        super(GraphFrequency, self).__init__(data, xname=name, yname=yname, order=order, dropna=dropna)

    def draw(self):

        freq = self._data.percents if self._percent else self._data.counts
        labels = self._data.categories.tolist()
        nbars = tuple(range(1, len(freq) + 1))
        grid_props = dict(linestyle='-', which='major', color='grey', alpha=0.75)

        # Create the figure and axes
        f, ax = subplots(figsize=(self._xsize, self._ysize))

        # Set the title
        f.suptitle(self._title, fontsize=14)

        # Create the graph, grid and labels
        if self._vertical:
            ax.xaxis.grid(True, **grid_props)
            ax.barh(nbars, freq.tolist(), color=self.get_color(0), zorder=3)
            ax.set_xlabel(self._yname)
            ax.set_yticks(nbars)
            max_len = max([len(str(g)) for g in labels])
            offset = max_len / 5 * 0.09
            subplots_adjust(left=offset)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
        else:
            ax.yaxis.grid(True, **grid_props)
            ax.bar(nbars, freq.tolist(), color=self.get_color(0), zorder=3)
            ax.set_ylabel(self._yname)
            ax.set_xticks(nbars)
            if any([True if len(str(g)) > 10 else False for g in labels]) or len(nbars) > 5:
                xticks(rotation=60)
                max_len = max([len(str(g)) for g in labels])
                offset = max_len / 5 * 0.09
                subplots_adjust(bottom=offset)
            ax.set_xticklabels(labels)

        # Save the figure to disk or display
        if self._save_to:
            savefig(self._save_to)
            close(f)
        else:
            show()
        pass
