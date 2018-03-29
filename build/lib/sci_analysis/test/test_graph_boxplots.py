import unittest
import numpy as np
import pandas as pd
import scipy.stats as st
from os import path, getcwd
from warnings import catch_warnings, simplefilter

from ..graphs import GraphBoxplot
from .. data import Vector
from ..analysis.exc import NoDataError


class TestWarnings(unittest.TestCase):
    """A TestCase subclass with assertWarns substitute to cover python 2.7 which doesn't have an assertWarns method."""

    def assertWarnsCrossCompatible(self, expected_warning, *args, **kwargs):
        with catch_warnings(record=True) as warning_list:
            simplefilter('always')
            callable_obj = args[0]
            args = args[1:]
            callable_obj(*args, **kwargs)
            self.assertTrue(any(item.category == expected_warning for item in warning_list))


class MyTestCase(TestWarnings):

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

    def test_100_boxplot_2_default(self):
        """Generate a boxplot graph with default settings"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           save_to='{}test_box_100'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_101_boxplot_2_no_nqp(self):
        """Generate a boxplot graph with no nqp"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           nqp=False,
                           save_to='{}test_box_101'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_102_boxplot_2_weird_variance(self):
        """Generate a boxplot graph with small and large variance"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(0, 0.1, size=2000)
        input_2_array = st.norm.rvs(1, 8, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           save_to='{}test_box_102'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_103_boxplot_2_groups(self):
        """Generate a boxplot graph with set group names"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           groups=('Group 1', 'Group 2'),
                           save_to='{}test_box_103'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_104_boxplot_2_names_title(self):
        """Generate a boxplot graph with set xname, yname and title"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           xname='Test Groups',
                           yname='Test Data',
                           title='Title Test',
                           save_to='{}test_box_104'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_105_boxplot_2_diff_size(self):
        """Generate a boxplot graph with different sizes"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(0, 5, size=1234)
        input_2_array = st.norm.rvs(0, 5, size=56)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           title='Diff Size',
                           save_to='{}test_box_105'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_106_boxplot_2_diff_size_diff_disto(self):
        """Generate a boxplot graph with different sizes and different distributions"""
        np.random.seed(987654321)
        input_1_array = st.weibull_min.rvs(2, size=1234)
        input_2_array = st.norm.rvs(0, size=56)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           title='Diff Size, Diff Distribution',
                           save_to='{}test_box_106'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_107_boxplot_2_diff_size_diff_disto_dict(self):
        """Generate a boxplot graph with different sizes and different distributions as a dict"""
        np.random.seed(987654321)
        input_1_array = st.weibull_min.rvs(2, size=1234)
        input_2_array = st.norm.rvs(0, size=56)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot({'Group 1': input_1_array, 'Group 2': input_2_array},
                           title='Diff Size, Diff Distribution Dict',
                           save_to='{}test_box_107'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_108_boxplot_2_size_4(self):
        """Generate a boxplot graph with size 4"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(1, size=4)
        input_2_array = st.norm.rvs(size=4)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot({'Group 1': input_1_array, 'Group 2': input_2_array},
                           title='Size 4',
                           save_to='{}test_box_108'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_109_boxplot_2_at_min_size(self):
        """Generate a boxplot graph with size 2"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2)
        input_2_array = st.norm.rvs(size=3)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot({'Group 1': input_1_array, 'Group 2': input_2_array},
                           title='At Min Size',
                           save_to='{}test_box_109'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_110_boxplot_2_min_size(self):
        """Catch the min size case"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=1)
        input_2_array = st.norm.rvs(size=2)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array)])
        res = GraphBoxplot({'Group 1': input_1_array, 'Group 2': input_2_array},
                           title='Single point',
                           save_to='{}test_box_110'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median([np.median(input_1_array), np.median(input_2_array)]))

    def test_111_boxplot_2_missing_data(self):
        """Generate a boxplot with missing data"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        indicies_x = list(np.random.randint(0, 1999, 200))
        indicies_y = list(np.random.randint(0, 1999, 200))
        for i in indicies_x:
            input_1_array = np.insert(input_1_array, i, np.nan, axis=0)
        for i in indicies_y:
            input_2_array = np.insert(input_2_array, i, np.nan, axis=0)
        gmean = np.nanmean((np.nanmean(input_1_array), np.nanmean(input_2_array)))
        gmedian = np.nanmedian([np.nanmedian(input_1_array), np.nanmedian(input_2_array)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           title='Random Missing Data',
                           save_to='{}test_box_111'.format(self.save_path))
        self.assertTrue(res)
        self.assertAlmostEqual(gmean, res.grand_mean((np.nanmean(input_1_array), np.nanmean(input_2_array))), 4)
        self.assertAlmostEqual(gmedian, res.grand_median([np.nanmedian(input_1_array), np.nanmedian(input_2_array)]), 4)

    def test_112_boxplot_2_empty_arrays(self):
        """Catch the case where both arrays are empty"""
        np.random.seed(987654321)
        input_1_array = np.array([])
        input_2_array = np.array([])
        self.assertRaises(NoDataError, lambda: GraphBoxplot(input_1_array, input_2_array))

    def test_113_boxplot_2_empty_lists(self):
        """Catch the case where both lists are empty"""
        np.random.seed(987654321)
        input_1_array = []
        input_2_array = []
        self.assertRaises(NoDataError, lambda: GraphBoxplot(input_1_array, input_2_array))

    def test_114_boxplot_2_strings(self):
        """Generate a boxplot graph with 2 string lists"""
        np.random.seed(987654321)
        input_1_array = ["this", '2', 'is', '4.0', 'a', '6', 'string']
        input_2_array = ['3.0', "here's", '6', 'a', '9.0', 'string']
        gmean = np.mean((np.mean([2, 4, 6]), np.mean([3, 6, 9])))
        gmedian = np.median((np.median([2, 4, 6]), np.median([3, 6, 9])))
        res = GraphBoxplot(input_1_array, input_2_array,
                           title='String test',
                           save_to='{}test_box_114'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean([2, 4, 6]), np.mean([3, 6, 9]))))
        self.assertEqual(gmedian, res.grand_median((np.median([2, 4, 6]), np.median([3, 6, 9]))))

    def test_115_boxplot_2_2dim_array(self):
        """Generate a boxplot graph with 2 2dim arrays"""
        np.random.seed(987654321)
        input_1_array = np.array([[1, 2, 3], [4, 5, 6]])
        input_2_array = np.array([[3, 4, 5], [6, 7, 8]])
        gmean = np.nanmean((np.nanmean(input_1_array, axis=None), np.nanmean(input_2_array, axis=None)))
        gmedian = np.nanmedian([np.nanmedian(input_1_array, axis=None), np.nanmedian(input_2_array, axis=None)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           title='2dim Array',
                           save_to='{}test_box_115'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.nanmean(input_1_array, axis=None),
                                                np.nanmean(input_2_array, axis=None))))
        self.assertEqual(gmedian, res.grand_median([np.nanmedian(input_1_array, axis=None),
                                                    np.nanmedian(input_2_array, axis=None)]))

    def test_116_boxplot_2_3dim_array(self):
        """Generate a boxplot graph with 2 3dim arrays"""
        np.random.seed(987654321)
        input_1_array = np.array([[[1, 2, 3], [3, 4, 5]], [[6, 7, 8], [8, 9, 10]]])
        input_2_array = np.array([[[2, 3, 4], [5, 6, 7]], [[7, 8, 9], [10, 11, 12]]])
        gmean = np.nanmean((np.nanmean(input_1_array, axis=None), np.nanmean(input_2_array, axis=None)))
        gmedian = np.nanmedian([np.nanmedian(input_1_array, axis=None), np.nanmedian(input_2_array, axis=None)])
        res = GraphBoxplot(input_1_array, input_2_array,
                           title='3dim Array',
                           save_to='{}test_box_116'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.nanmean(input_1_array, axis=None),
                                                np.nanmean(input_2_array, axis=None))))
        self.assertEqual(gmedian, res.grand_median([np.nanmedian(input_1_array, axis=None),
                                                    np.nanmedian(input_2_array, axis=None)]))

    def test_117_boxplot_2_3dim_list(self):
        """Generate a boxplot graph with 2 3dim lists"""
        np.random.seed(987654321)
        input_1_array = [[['1', 'two', '3'], ['4', '5', 'six']], [['7', '8', '9'], ['ten', '11', '12']]]
        input_2_array = [[['one', '2', '3'], ['four', '5', '6']], [['7', '8', '9'], ['ten', '11', '12']]]
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array,
                                     title='String Array Test',
                                     save_to='{}test_box_117'.format(self.save_path)))

    def test_118_boxplot_4_default(self):
        """Generate a boxplot graph with 4 arrays and default settings"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array), np.mean(input_3_array),
                         np.mean(input_4_array)))
        gmedian = np.median([np.median(input_1_array), np.median(input_2_array), np.median(input_3_array),
                             np.median(input_4_array)])
        res = GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                           save_to='{}test_box_118'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, np.mean((np.mean(input_1_array), np.mean(input_2_array),
                                         np.mean(input_3_array), np.mean(input_4_array))))
        self.assertEqual(gmedian, np.median((np.median(input_1_array), np.median(input_2_array),
                                             np.median(input_3_array), np.median(input_4_array))))

    def test_119_boxplot_4_no_nqp(self):
        """Generate a boxplot graph with 4 arrays and no nqp"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array),
                         np.mean(input_3_array), np.mean(input_4_array)))
        gmedian = np.median((np.median(input_1_array), np.median(input_2_array),
                             np.median(input_3_array), np.median(input_4_array)))
        res = GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                           nqp=False,
                           save_to='{}test_box_119'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, np.mean((np.mean(input_1_array), np.mean(input_2_array),
                                         np.mean(input_3_array), np.mean(input_4_array))))
        self.assertEqual(gmedian, np.median((np.median(input_1_array), np.median(input_2_array),
                                             np.median(input_3_array), np.median(input_4_array))))

    def test_120_boxplot_4_no_nqp_groups(self):
        """Generate a boxplot graph with 4 arrays, no nqp and set groups"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                                     nqp=False,
                                     groups=('Group 1', 'Group 2', 'Group 3', 'Group 4'),
                                     save_to='{}test_box_120'.format(self.save_path)))

    def test_121_boxplot_4_no_nqp_dict(self):
        """Generate a boxplot graph with 4 arrays from a dict and no nqp"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        self.assertTrue(GraphBoxplot({'Group 1': input_1_array, 'Group 2': input_2_array, 'Group 3': input_3_array,
                                      'Group 4': input_4_array},
                                     nqp=True,
                                     save_to='{}test_box_121'.format(self.save_path)))

    def test_122_boxplot_4_empty_array(self):
        """Generate a boxplot graph with 1 empty array"""
        np.random.seed(987654321)
        # TODO: Note in the documentation that if an array is ignored this way, the auto-number isn't skipped now.
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = np.array([])
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                                     title='1 Missing Array',
                                     save_to='{}test_box_122'.format(self.save_path)))

    def test_123_boxplot_4_2_empty_arrays(self):
        """Generate a boxplot graph with 2 empty arrays"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = np.array([])
        input_3_array = []
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                                     title='2 Missing Arrays',
                                     save_to='{}test_box_123'.format(self.save_path)))

    def test_124_boxplot_4_all_empty(self):
        """Catch the case where all arrays are empty"""
        np.random.seed(987654321)
        input_1_array = ['this', 'is', 'an', 'array']
        input_2_array = ['this', 'is', 'another', 'array']
        input_3_array = ['this', 'is', 'not', 'the', 'array', "you're", 'looking', 'for']
        input_4_array = ['and', 'nope']
        self.assertTrue(NoDataError, lambda: GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array))

    def test_125_boxplot_4_strings(self):
        """Generate a boxplot graph from strings"""
        np.random.seed(987654321)
        input_1_array = ["this", '2', 'is', '4.0', 'a', '6', 'string']
        input_2_array = ['3.0', "here's", '6', 'a', '9.0', 'string']
        input_3_array = ['1', '2', '2', 'two', '3', '3', '3', '4']
        input_4_array = ['4', '4', 'four', '4', 'five', '1']
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                                     title='4 Arrays Strings',
                                     save_to='{}test_box_125'.format(self.save_path)))

    def test_126_boxplot_14_default(self):
        """Generate a boxplot graph with 14 arrays"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=1847)
        input_3_array = st.norm.rvs(0.5, 0.5, size=1134)
        input_4_array = st.norm.rvs(0, 0.1, size=962)
        input_5_array = st.weibull_min.rvs(1.2, size=2000)
        input_6_array = st.norm.rvs(size=82)
        input_7_array = st.norm.rvs(0, 2, size=823)
        input_8_array = st.norm.rvs(2, size=2000)
        input_9_array = st.weibull_min.rvs(2, size=1200)
        input_10_array = st.norm.rvs(0.5, 1.5, size=200)
        input_11_array = st.norm.rvs(-1, size=1732)
        input_12_array = st.norm.rvs(-0.5, 2, size=1386)
        input_13_array = st.norm.rvs(0, 0.5, size=548)
        input_14_array = st.weibull_min.rvs(1.7, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array), np.mean(input_3_array), np.mean(input_4_array),
                         np.mean(input_5_array), np.mean(input_6_array), np.mean(input_7_array), np.mean(input_8_array),
                         np.mean(input_9_array), np.mean(input_10_array), np.mean(input_11_array),
                         np.mean(input_12_array), np.mean(input_13_array), np.mean(input_14_array)))
        gmedian = np.median((np.median(input_1_array), np.median(input_2_array), np.median(input_3_array),
                             np.median(input_4_array), np.median(input_5_array), np.median(input_6_array),
                             np.median(input_7_array), np.median(input_8_array), np.median(input_9_array),
                             np.median(input_10_array), np.median(input_11_array), np.median(input_12_array),
                             np.median(input_13_array), np.median(input_14_array)))
        res = GraphBoxplot({'Group 1': input_1_array,
                            'Group 2': input_2_array,
                            'Group 3': input_3_array,
                            'Group 4': input_4_array,
                            'Group 5': input_5_array,
                            'Group 6': input_6_array,
                            'Group 7': input_7_array,
                            'Group 8': input_8_array,
                            'Group 9': input_9_array,
                            'Group 10': input_10_array,
                            'Group 11': input_11_array,
                            'Group 12': input_12_array,
                            'Group 13': input_13_array,
                            'Group 14': input_14_array},
                           title='14 Arrays',
                           save_to='{}test_box_126'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array), np.mean(input_3_array),
                                                np.mean(input_4_array), np.mean(input_5_array), np.mean(input_6_array),
                                                np.mean(input_7_array), np.mean(input_8_array), np.mean(input_9_array),
                                                np.mean(input_10_array), np.mean(input_11_array),
                                                np.mean(input_12_array), np.mean(input_13_array),
                                                np.mean(input_14_array))))
        self.assertEqual(gmedian, res.grand_median((np.median(input_1_array), np.median(input_2_array),
                                                    np.median(input_3_array), np.median(input_4_array),
                                                    np.median(input_5_array), np.median(input_6_array),
                                                    np.median(input_7_array), np.median(input_8_array),
                                                    np.median(input_9_array), np.median(input_10_array),
                                                    np.median(input_11_array), np.median(input_12_array),
                                                    np.median(input_13_array), np.median(input_14_array))))

    def test_127_boxplot_1_default(self):
        """Generate a boxplot graph with 1 array"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(1, size=2000)
        self.assertTrue(GraphBoxplot(input_1_array,
                                     title='1 Array',
                                     save_to='{}test_box_127'.format(self.save_path)))

    def test_128_boxplot_1_no_nqp(self):
        """Generate a boxplot graph with 1 array and no nqp"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphBoxplot(input_1_array,
                                     nqp=False,
                                     title='1 Array no NQP',
                                     save_to='{}test_box_128'.format(self.save_path)))

    def test_129_boxplot_1_groups(self):
        """Generate a boxplot graph with 1 array and set groups"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphBoxplot(input_1_array,
                                     groups=['Group 1'] * 2000,
                                     title='1 Array Groups Set',
                                     save_to='{}test_box_129'.format(self.save_path)))

    def test_130_boxplot_1_dict(self):
        """Generate a boxplot graph with 1 array from a dict"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphBoxplot({'Group 1': input_1_array},
                                     title='1 Array Dict',
                                     save_to='{}test_box_130'.format(self.save_path)))

    def test_131_boxplot_1_no_data(self):
        """Catch the case where the 1 and only array is empty"""
        np.random.seed(987654321)
        input_1_array = np.array([])
        self.assertRaises(NoDataError, lambda: GraphBoxplot(input_1_array))

    def test_132_boxplot_4_missing_3(self):
        """Generate a boxplot graph with 4 arrays where 3 are missing"""
        np.random.seed(987654321)
        input_1_array = np.array([])
        input_2_array = ['One', 'two', 'three', 'four']
        input_3_array = st.norm.rvs(size=5)
        input_4_array = []
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                                     title='4 Array 3 Missing',
                                     save_to='{}test_box_132'.format(self.save_path)))

    def test_133_boxplot_horizontal_labels_length_size(self):
        """Generate a boxplot graph at the max horizontal labels"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        input_3_array = st.norm.rvs(size=100)
        input_4_array = st.norm.rvs(size=100)
        input_5_array = st.norm.rvs(size=100)
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array, input_5_array,
                                     title='Horizontal labels test',
                                     groups=['1111111111', '2222222222', '3333333333', '4444444444', '5555555555'],
                                     save_to='{}test_box_133'.format(self.save_path)))

    def test_134_boxplot_vertical_labels_length(self):
        """Generate a boxplot graph with vertical labels"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        input_3_array = st.norm.rvs(size=100)
        input_4_array = st.norm.rvs(size=100)
        input_5_array = st.norm.rvs(size=100)
        input_6_array = st.norm.rvs(size=100)
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array, input_5_array,
                                     input_6_array,
                                     title='Horizontal labels test',
                                     groups=['1111111111', '2222222222', '3333333333', '4444444444', '5555555555',
                                             '6666666666'],
                                     save_to='{}test_box_134'.format(self.save_path)))

    def test_135_boxplot_vertical_labels_size(self):
        """Generate a boxplot graph with vertical labels"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        input_3_array = st.norm.rvs(size=100)
        input_4_array = st.norm.rvs(size=100)
        self.assertTrue(GraphBoxplot(input_1_array, input_2_array, input_3_array, input_4_array,
                                     title='Horizontal labels test',
                                     groups=['1234567890a', '1234567890b', '1234567890c', '1234567890d'],
                                     save_to='{}test_box_135'.format(self.save_path)))

    def test_136_boxplot_4_groups_5(self):
        """Generate a boxplot graph with 4 arrays and 5 groups"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        input_3_array = st.norm.rvs(size=100)
        input_4_array = st.norm.rvs(size=100)
        self.assertRaises(AttributeError, lambda: GraphBoxplot(
            input_1_array,
            input_2_array,
            input_3_array,
            input_4_array,
            groups=['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'],
            title='4 Arrays 5 Groups',
            save_to='{}test_box_136'.format(self.save_path)))

    def test_137_boxplot_4_groups_3(self):
        """Generate a boxplot graph with 4 arrays and 3 groups"""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        input_3_array = st.norm.rvs(size=100)
        input_4_array = st.norm.rvs(size=100)
        self.assertRaises(AttributeError, lambda: GraphBoxplot(input_1_array,
                                                               input_2_array,
                                                               input_3_array,
                                                               input_4_array,
                                                               groups=['Group 1', 'Group 2', 'Group 3']))

    def test_138_boxplot_vector(self):
        """Generate a boxplot graph from a Vector object."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        gmean = np.mean((np.mean(input_1_array), np.mean(input_2_array)))
        gmedian = np.median((np.median(input_1_array), np.median(input_2_array)))
        vector = Vector(input_1_array).append(Vector(input_2_array))
        res = GraphBoxplot(vector, title='Vector Simple Test', save_to='{}test_box_138'.format(self.save_path))
        self.assertTrue(res)
        self.assertEqual(gmean, res.grand_mean((np.mean(input_1_array), np.mean(input_2_array))))
        self.assertEqual(gmedian, res.grand_median((np.median(input_1_array), np.median(input_2_array))))

    def test_139_boxplot_vector_ignore_groups(self):
        """Generate a boxplot graph from a Vector object which should ignore the groups kwargs."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        vector = Vector(input_1_array).append(Vector(input_2_array))
        self.assertTrue(GraphBoxplot(vector,
                                     title='Vector Simple Test',
                                     groups=('Group 1', 'Group 2'),
                                     save_to='{}test_box_139'.format(self.save_path)))

    def test_140_boxplot_vector_with_group_names(self):
        """Generate a boxplot graph from a Vector object with specified group names."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        vector = (Vector(input_1_array, groups=['Group 1'] * 2000)
                  .append(Vector(input_2_array, groups=['Group 2'] * 2000)))
        self.assertTrue(GraphBoxplot(vector,
                                     title='Vector Simple Test',
                                     save_to='{}test_box_140'.format(self.save_path)))

    def test_141_boxplot_vector_4_default(self):
        """Generate a boxplot graph from a vector object with four groups."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        vector = (Vector(input_1_array)
                  .append(Vector(input_2_array))
                  .append(Vector(input_3_array))
                  .append(Vector(input_4_array)))
        self.assertTrue(GraphBoxplot(vector, save_to='{}test_box_141'.format(self.save_path)))

    def test_142_boxplot_vector_with_groups_4_default(self):
        """Generate a boxplot graph from a vector object with four groups."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        vector = (Vector(input_1_array, groups=['Group 1'] * 2000)
                  .append(Vector(input_2_array, groups=['Group 2'] * 2000))
                  .append(Vector(input_3_array, groups=['Group 3'] * 2000))
                  .append(Vector(input_4_array, groups=['Group 4'] * 2000)))
        self.assertTrue(GraphBoxplot(vector, save_to='{}test_box_142'.format(self.save_path)))

    def test_143_boxplot_from_columns_default(self):
        """Generate a boxplot graph from a single column with group column."""
        np.random.seed(987654321)
        input_1_array = pd.DataFrame({'input': st.norm.rvs(size=2000), 'group': ['Group 1'] * 2000})
        input_2_array = pd.DataFrame({'input': st.norm.rvs(1, size=2000), 'group': ['Group 2'] * 2000})
        df = pd.concat([input_1_array, input_2_array])
        self.assertTrue(GraphBoxplot(df['input'], groups=df['group'],
                                     title='DataFrame Simple Test',
                                     save_to='{}test_box_143'.format(self.save_path)))

    def test_144_boxplot_from_columns_with_groups_4_default(self):
        """Generate a boxplot graph from a single column with group column."""
        np.random.seed(987654321)
        input_1_array = pd.DataFrame({'input': st.norm.rvs(size=2000), 'group': ['Group 1'] * 2000})
        input_2_array = pd.DataFrame({'input': st.norm.rvs(1, size=2000), 'group': ['Group 2'] * 2000})
        input_3_array = pd.DataFrame({'input': st.norm.rvs(2, 0.5, size=2000), 'group': ['Group 3'] * 2000})
        input_4_array = pd.DataFrame({'input': st.weibull_min.rvs(1.4, size=2000), 'group': ['Group 4'] * 2000})
        df = pd.concat([input_1_array, input_2_array, input_3_array, input_4_array])
        self.assertTrue(GraphBoxplot(df['input'], groups=df['group'],
                                     title='DataFrame 4 Groups',
                                     save_to='{}test_box_144'.format(self.save_path)))

    def test_145_boxplot_data_column_length_unequal_to_group_column_length(self):
        """Check the case where the length of the data array doesn't match the length of the group labels array."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        self.assertRaises(AttributeError, lambda: GraphBoxplot(input_1_array, groups=['Group 1']))

    # def test_146_boxplot_issues_depricated_warning(self):
    #     """Check to make sure a Deprication warnings is raised if passing in multiple arguments."""
    #     np.random.seed(987654321)
    #     input_1_array = st.norm.rvs(size=2000)
    #     input_2_array = st.norm.rvs(1, size=2000)
    #     self.assertWarnsCrossCompatible(FutureWarning,
    #                                     lambda: GraphBoxplot(input_1_array, input_2_array,
    #                                                          title='Raise Warning',
    #                                                          save_to='{}test_box_146'.format(self.save_path)))

    def test_147_boxplot_scalar(self):
        """Generate a boxplot from a scalar value."""
        input_1_array = 3
        self.assertTrue(GraphBoxplot(input_1_array, title='Scalar Boxplot',
                                     save_to='{}test_box_147'.format(self.save_path)))

    def test_148_boxplot_vector_no_circles(self):
        """Generate a boxplot graph from a vector object with four groups and no circles."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        vector = (Vector(input_1_array)
                  .append(Vector(input_2_array))
                  .append(Vector(input_3_array))
                  .append(Vector(input_4_array)))
        self.assertTrue(GraphBoxplot(vector, save_to='{}test_box_148'.format(self.save_path), circles=False))

    def test_149_no_gmean(self):
        """Generate a boxplot graph from a vector object with four groups and no grand mean line."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        vector = (Vector(input_1_array)
                  .append(Vector(input_2_array))
                  .append(Vector(input_3_array))
                  .append(Vector(input_4_array)))
        res = GraphBoxplot(vector,
                           gmean=False,
                           save_to='{}test_box_149'.format(self.save_path))
        self.assertTrue(res)

    def test_150_no_gmedian(self):
        """Generate a boxplot graph from a vector object with four groups and no grand median line."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        vector = (Vector(input_1_array)
                  .append(Vector(input_2_array))
                  .append(Vector(input_3_array))
                  .append(Vector(input_4_array)))
        res = GraphBoxplot(vector,
                           gmedian=False,
                           save_to='{}test_box_150'.format(self.save_path))
        self.assertTrue(res)

    def test_151_no_gmedian_or_gmean(self):
        """Generate a boxplot graph from a vector object with four groups and no grand mean or median line."""
        np.random.seed(987654321)
        input_1_array = st.norm.rvs(size=2000)
        input_2_array = st.norm.rvs(1, size=2000)
        input_3_array = st.norm.rvs(2, 0.5, size=2000)
        input_4_array = st.weibull_min.rvs(1.4, size=2000)
        vector = (Vector(input_1_array)
                  .append(Vector(input_2_array))
                  .append(Vector(input_3_array))
                  .append(Vector(input_4_array)))
        res = GraphBoxplot(vector,
                           gmean=False,
                           gmedian=False,
                           save_to='{}test_box_151'.format(self.save_path))
        self.assertTrue(res)


if __name__ == '__main__':
    unittest.main()
