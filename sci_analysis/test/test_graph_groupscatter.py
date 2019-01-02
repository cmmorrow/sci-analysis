import unittest
import numpy as np
import pandas as pd
import scipy.stats as st
from os import path, getcwd

from ..graphs import GraphGroupScatter
from ..data import Vector
from ..analysis.exc import NoDataError
from ..data import UnequalVectorLengthError


class MyTestCase(unittest.TestCase):

    @property
    def save_path(self):
        if getcwd().split('/')[-1] == 'test':
            return './images/'
        elif getcwd().split('/')[-1] == 'sci_analysis':
            if path.exists('./setup.py'):
                return './sci_analysis/test/images/'
            else:
                return './test/images/'
        else:
            './'

    def test_1_scatter_two_groups_default(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_1'.format(self.save_path)))

    def test_2_scatter_two_groups_no_fit(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], fit=False,
                                          save_to='{}test_group_scatter_2'.format(self.save_path)))

    def test_3_scatter_two_groups_no_points(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], points=False,
                                          save_to='{}test_group_scatter_3'.format(self.save_path)))

    def test_4_scatter_two_groups_highlight_one(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight=[2],
                                          save_to='{}test_group_scatter_4'.format(self.save_path)))

    def test_5_scatter_three_groups_highlight_two(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = [1] * 100 + [2] * 100 + [3] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight=[2, 3],
                                          save_to='{}test_group_scatter_5'.format(self.save_path)))

    def test_6_scatter_two_groups_highlight_one_no_points(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight=[2],
                                          points=False, save_to='{}test_group_scatter_6'.format(self.save_path)))

    def test_7_scatter_two_groups_highlight_one_no_fit(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight=[2],
                                          fit=False, save_to='{}test_group_scatter_7'.format(self.save_path)))

    def test_8_scatter_two_groups_highlight_one_scalar_num(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight=2,
                                          save_to='{}test_group_scatter_8'.format(self.save_path)))

    def test_9_scatter_two_groups_string_names_highlight_one(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = ['a'] * 100 + ['b'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight=['b'],
                                          save_to='{}test_group_scatter_9'.format(self.save_path)))

    def test_10_scatter_three_groups_string_names_highlight_scalar_string(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], highlight='bc',
                                          save_to='{}test_group_scatter_10'.format(self.save_path)))

    def test_11_scatter_three_groups_invalid_highlight_groups(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          highlight=['z', 'y', 'x'],
                                          save_to='{}test_group_scatter_11'.format(self.save_path)))

    def test_12_scatter_two_groups_no_boxplot_borders(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = ['a'] * 100 + ['b'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          boxplot_borders=False,
                                          save_to='{}test_group_scatter_12'.format(self.save_path)))

    def test_13_scatter_two_groups_title(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = ['a'] * 100 + ['b'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          title='Title Test', save_to='{}test_group_scatter_13'.format(self.save_path)))

    def test_14_scatter_two_groups_labels(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = ['a'] * 100 + ['b'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'], xname='Test x',
                                          yname='Test y', save_to='{}test_group_scatter_14'.format(self.save_path)))

    def test_15_scatter_three_groups_auto_named(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'],
                                          save_to='{}test_group_scatter_15'.format(self.save_path)))

    def test_16_scatter_one_group_default(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        grp = ['a'] * 100
        input_array = pd.DataFrame({'a': input_1_x, 'b': input_1_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_16'.format(self.save_path)))

    def test_17_scatter_three_groups_vector_input_default(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = Vector(cs_x, other=cs_y, groups=grp)
        self.assertTrue(GraphGroupScatter(input_array, save_to='{}test_group_scatter_17'.format(self.save_path)))

    def test_18_scatter_three_groups_vector_input_highlight_one(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = Vector(cs_x, other=cs_y, groups=grp)
        self.assertTrue(GraphGroupScatter(input_array, highlight=['b'],
                                          save_to='{}test_group_scatter_18'.format(self.save_path)))

    def test_19_scatter_one_group_matplotlib_bug(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=3)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        grp = ['a'] * 3
        input_array = pd.DataFrame({'a': input_1_x, 'b': input_1_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_19'.format(self.save_path)))

    def test_20_scatter_two_groups_matplotlib_bug(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=4)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 4 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_20'.format(self.save_path)))

    def test_21_scatter_two_groups_unequal_x_and_y_size(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x if x > 0.0]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x if x > 0.0]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        self.assertRaises(UnequalVectorLengthError, lambda: GraphGroupScatter(cs_x, cs_y, groups=grp))

    def test_22_scatter_two_groups_wrong_group_size(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1, 2]
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        self.assertRaises(UnequalVectorLengthError, lambda: GraphGroupScatter(cs_x, cs_y, groups=grp))

    def test_23_no_data(self):
        """Test the case where there's no data."""
        self.assertRaises(NoDataError, lambda: GraphGroupScatter([], []))

    def test_24_scatter_three_groups_different_sizes(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=1)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=10)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 1 + ['b'] * 10 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_24'.format(self.save_path)))

    def test_25_scatter_two_groups_no_ydata(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertRaises(AttributeError, lambda: GraphGroupScatter(input_array['a'], groups=input_array['c']))

    def test_26_scatter_three_groups_long_group_names(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=100)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=100)[0] for x in input_3_x]) - 0.5
        grp = ['11111111111111111111'] * 100 + ['222222222222222222222'] * 100 + ['3333333333333333333333'] * 100
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_26'.format(self.save_path)))

    def test_27_scatter_two_groups_negative_corr(self):
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [2 - (x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(GraphGroupScatter(input_array['a'], input_array['b'], groups=input_array['c'],
                                          save_to='{}test_group_scatter_27'.format(self.save_path)))

    def test_28_scatter_two_groups_labels(self):
        """Test the case where labels are provided."""
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_labels_array = np.random.choice(list('ABCDE'), size=200)
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(
            GraphGroupScatter(
                input_array['a'],
                input_array['b'],
                groups=input_array['c'],
                labels=input_labels_array,
                highlight=['A'],
                save_to='{}test_group_scatter_28'.format(self.save_path)
            )
        )

    def test_29_scatter_two_groups_labels_and_group_highlight(self):
        """Test the case where labels and groups are highlighted."""
        np.random.seed(987654321)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_labels_array = np.random.choice(list('ABCDE'),  size=220)
        grp = [1] * 110 + [2] * 110
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        indicies_x = list(np.random.randint(0, 199, 20))
        indicies_y = list(np.random.randint(0, 199, 20))
        for i in indicies_x:
            cs_x = np.insert(cs_x, i, np.nan, axis=0)
        for i in indicies_y:
            cs_y = np.insert(cs_y, i, np.nan, axis=0)
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertTrue(
            GraphGroupScatter(
                input_array['a'],
                input_array['b'],
                groups=input_array['c'],
                labels=input_labels_array,
                highlight=[1],
                save_to='{}test_group_scatter_29'.format(self.save_path)
            )
        )

    def test_30_groupscatter_dataframe(self):
        """Tests graphscater with dataframe input."""
        np.random.seed(987654321)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['labels'] = np.random.choice(list('ABCDE'), len(df)).tolist()
        df['groups'] = np.random.choice(list('XYZ'), len(df)).tolist()
        self.assertTrue(
            GraphGroupScatter(
                df['x'],
                df['y'],
                groups=df['groups'],
                labels=df['labels'],
                highlight=['A'],
                save_to='{}test_group_scatter_30'.format(self.save_path)
            )
        )

    def test_31_groupscatter_labels_no_highlight(self):
        """Test the case where labels are given, but no highlights specified."""
        np.random.seed(987654321)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['labels'] = np.random.choice(list('ABCDE'), len(df)).tolist()
        df['groups'] = np.random.choice(list('XYZ'), len(df)).tolist()
        self.assertTrue(
            GraphGroupScatter(
                df['x'],
                df['y'],
                groups=df['groups'],
                labels=df['labels'],
                save_to='{}test_group_scatter_31'.format(self.save_path)
            )
        )

    def test_32_groupscatter_labels_invalid_labels_and_groups(self):
        """Test the case where all the highlights are not in groups or labels."""
        np.random.seed(987654321)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['labels'] = np.random.choice(list('ABCDE'), len(df)).tolist()
        df['groups'] = np.random.choice(list('XYZ'), len(df)).tolist()
        self.assertTrue(
            GraphGroupScatter(
                df['x'],
                df['y'],
                groups=df['groups'],
                labels=df['labels'],
                highlight=['XX', 2, 34],
                save_to='{}test_group_scatter_32'.format(self.save_path)
            )
        )

    def test_33_groupscatter_labels_individual_highlight(self):
        """Test the case where individual points are highlighted with one overrunning."""
        np.random.seed(987654321)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['labels'] = np.random.randint(10000, 50000, size=100)
        df['groups'] = np.random.choice(list('XYZ'), len(df)).tolist()
        self.assertTrue(
            GraphGroupScatter(
                df['x'],
                df['y'],
                groups=df['groups'],
                labels=df['labels'],
                highlight=df[df['x'] > 2]['labels'].tolist(),
                save_to='{}test_group_scatter_33'.format(self.save_path)
            )
        )

    def test_34_groupscatter_labels_individual_no_borders(self):
        """Test the case where individual points are highlighted with one overrunning and no boxplot borders."""
        np.random.seed(987654321)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['labels'] = np.random.randint(10000, 50000, size=100)
        df['groups'] = np.random.choice(list('XYZ'), len(df)).tolist()
        self.assertTrue(
            GraphGroupScatter(
                df['x'],
                df['y'],
                boxplot_borders=False,
                groups=df['groups'],
                labels=df['labels'],
                highlight=df[df['x'] > 2]['labels'].tolist(),
                save_to='{}test_group_scatter_34'.format(self.save_path)
            )
        )


if __name__ == '__main__':
    unittest.main()
