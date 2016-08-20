

class DefaultPreferences(type):
    """The type for Default Preferences that cannot be modified"""

    def __setattr__(cls, key, value):
        if key == "defaults":
            raise AttributeError("Cannot override defaults")
        else:
            return type.__setattr__(cls, key, value)

    def __delattr__(cls, item):
        if item == "defaults":
            raise AttributeError("Cannot delete defaults")
        else:
            return type.__delattr__(cls, item)


class Preferences(object):
    """The base Preferences class"""

    __metaclass__ = DefaultPreferences

    def list(self):
        print(self.__dict__)
        return self.__dict__

    def defaults(self):
        return tuple(self.__dict__.values())


class GraphPreferences(object):
    """Handles graphing preferences."""

    class Plot(object):
        boxplot = True
        histogram = True
        cdf = False
        oneway = True
        probplot = True
        scatter = True
        tukey = False
        histogram_borders = False
        boxplot_borders = False
        defaults = (boxplot, histogram, cdf, oneway, probplot, scatter, tukey, histogram_borders, boxplot_borders)

    distribution = {'counts': False,
                    'violin': False,
                    'boxplot': True,
                    'fit': False,
                    'fit_style': 'r--',
                    'fit_width': '2',
                    'cdf_style': 'k-',
                    'distribution': 'norm',
                    'bins': 20,
                    'color': 'green'
                    }

    bivariate = {'points': True,
                 'point_style': 'k.',
                 'contours': False,
                 'contour_width': 1.25,
                 'fit': True,
                 'fit_style': 'r-',
                 'fit_width': 1,
                 'boxplot': True,
                 'violin': True,
                 'bins': 20,
                 'color': 'green'
                 }

    oneway = {'boxplot': True,
              'violin': False,
              'point_style': '^',
              'line_style': '-'
              }
