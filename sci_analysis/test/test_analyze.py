import unittest
import numpy as np
import pandas as pd
import scipy.stats as st

from ..analysis.exc import NoDataError
from ..analysis import analyze, analyse
from .base import TestWarnings


class MyTestCase(TestWarnings):

    def test_100_catch_no_data_1_array(self):
        """Catch the case where no data is passed"""
        self.assertRaises(NoDataError, lambda: analyze([]))

    def test_101_catch_no_data_None(self):
        """Catch the case where None is passed"""
        self.assertRaises(ValueError, lambda: analyze(None))

    def test_102_catch_xdata_no_iterable(self):
        """Catch the case where xdata is not iterable"""
        self.assertRaises(TypeError, lambda: analyze(1))

    def test_104_ttest_large_default(self):
        """Perform an analysis on a large sample using the ttest"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array], debug=True,
                                 save_to='{}test_analyze_104'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_105_ttest_small_default(self):
        """Perform an analysis on a small sample using the ttest"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=10)
        input_2_array = st.norm.rvs(size=10)
        self.assertEqual(analyze([input_1_array, input_2_array], debug=True,
                                 save_to='{}test_analyze_105'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_106_ttest_large_group(self):
        """Perform an analysis on a large sample using the ttest with set group names"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 groups=['Test 1', 'Test 2'],
                                 debug=True,
                                 save_to='{}test_analyze_106'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_107_ttest_large_dict(self):
        """Perform an analysis on a large sample using the ttest with set dict"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze({'dTest 1': input_1_array, 'dTest 2': input_2_array},
                                 debug=True,
                                 save_to='{}test_analyze_107'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_108_ttest_xlabel_ylabel(self):
        """Perform an analysis on a large sample using the ttest with labels set"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 title='Labels test',
                                 xname='X Test',
                                 yname='Y Test',
                                 debug=True,
                                 save_to='{}test_analyze_108'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_109_mannwhitney_default(self):
        """Perform an analysis on a non-normal data set using the Mann Whitney test"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.weibull_min.rvs(1.2, size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 title='MannWhitney Default',
                                 debug=True,
                                 save_to='{}test_analyze_109'.format(self.save_path)),
                         ['Oneway', 'MannWhitney'])

    def test_110_mannwhitney_groups(self):
        """Perform an analysis on a non-normal data set using the Mann Whitney test"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.weibull_min.rvs(1.2, size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 groups=['Test 1', 'Test 2'],
                                 title='MannWhitney Groups',
                                 debug=True,
                                 save_to='{}test_analyze_110'.format(self.save_path)),
                         ['Oneway', 'MannWhitney'])

    def test_111_mannwhitney_groups(self):
        """Perform an analysis on a non-normal data set using the Mann Whitney test"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.weibull_min.rvs(1.2, size=100)
        self.assertEqual(analyze({'dTest 1': input_1_array, 'dTest 2': input_2_array},
                                 title='MannWhitney Dict',
                                 debug=True,
                                 save_to='{}test_analyze_111'.format(self.save_path)),
                         ['Oneway', 'MannWhitney'])

    def test_112_twosampleks_default(self):
        """Perform an analysis on a small bi-modal data set using the twosample ks test"""
        np.random.seed(self._seed)
        input_1_array = np.append(st.norm.rvs(0, 1, size=10), st.norm.rvs(10, 1, size=10))
        input_2_array = np.append(st.norm.rvs(0, 1, size=10), st.norm.rvs(10, 1, size=10))
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 title='TwoSampleKSTest Default',
                                 debug=True,
                                 save_to='{}test_analyze_112'.format(self.save_path)),
                         ['Oneway', 'TwoSampleKSTest'])

    def test_113_twosampleks_groups(self):
        """Perform an analysis on a small bi-modal data set using the twosample ks test"""
        np.random.seed(self._seed)
        input_1_array = np.append(st.norm.rvs(0, 1, size=10), st.norm.rvs(10, 1, size=10))
        input_2_array = np.append(st.norm.rvs(0, 1, size=10), st.norm.rvs(10, 1, size=10))
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 groups=['Group 1', 'Group 2'],
                                 title='TwoSampleKSTest Groups',
                                 debug=True,
                                 save_to='{}test_analyze_113'.format(self.save_path)),
                         ['Oneway', 'TwoSampleKSTest'])

    def test_114_twosampleks_dict(self):
        """Perform an analysis on a small bi-modal data set using the twosample ks test"""
        np.random.seed(self._seed)
        input_1_array = np.append(st.norm.rvs(0, 1, size=10), st.norm.rvs(10, 1, size=10))
        input_2_array = np.append(st.norm.rvs(0, 1, size=10), st.norm.rvs(10, 1, size=10))
        self.assertEqual(analyze({'dGroup 1': input_1_array, 'dGroup 2': input_2_array},
                                 title='TwoSampleKSTest Dict',
                                 debug=True,
                                 save_to='{}test_analyze_114'.format(self.save_path)),
                         ['Oneway', 'TwoSampleKSTest'])

    def test_115_ttest_name_categories_default(self):
        """Perform an analysis on a large sample using the ttest with labels set"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 title='Labels test 2',
                                 categories='X Test',
                                 name='Y Test',
                                 debug=True,
                                 save_to='{}test_analyze_115'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_116_ttest_name_categories_groups(self):
        """Perform an analysis on a large sample using the ttest with labels set"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 groups=['Group 1', 'Group 2'],
                                 title='Labels test 2 Groups',
                                 categories='X Test',
                                 name='Y Test',
                                 debug=True,
                                 save_to='{}test_analyze_116'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_117_ttest_name_categories_dict(self):
        """Perform an analysis on a large sample using the ttest with labels set"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze({'dGroup 1': input_1_array, 'dGroup 2': input_2_array},
                                 title='Labels test Dict',
                                 categories='X Test',
                                 name='Y Test',
                                 debug=True,
                                 save_to='{}test_analyze_117'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_118_ttest_alpha(self):
        """Perform an analysis on a large sample using the ttest with alpha 0.02"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 title='Alpha 0.02',
                                 alpha=0.02,
                                 debug=True,
                                 save_to='{}test_analyze_118'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_119_ttest_no_nqp(self):
        """Perform an analysis on a large sample using the ttest without a nqp"""
        np.random.seed(self._seed)
        input_1_array = st.norm.rvs(size=100)
        input_2_array = st.norm.rvs(size=100)
        self.assertEqual(analyze([input_1_array, input_2_array],
                                 title='No NQP',
                                 nqp=False,
                                 debug=True,
                                 save_to='{}test_analyze_119'.format(self.save_path)),
                         ['Oneway', 'TTest'])

    def test_120_bivariate_default(self):
        """Perform a correlation on two data sets with default settings"""
        np.random.seed(self._seed)
        input_x_array = st.weibull_min.rvs(2, size=200)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertEqual(analyze(input_x_array, input_y_array,
                                 debug=True,
                                 save_to='{}test_analyze_120'.format(self.save_path)),
                         ['Bivariate'])

    def test_121_bivariate_xname_yname(self):
        """Perform a correlation on two data sets with labels set"""
        np.random.seed(self._seed)
        input_x_array = st.weibull_min.rvs(2, size=200)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertEqual(analyze(input_x_array, input_y_array,
                                 xname='X Test',
                                 yname='Y Test',
                                 title='Labels Test',
                                 debug=True,
                                 save_to='{}test_analyze_121'.format(self.save_path)),
                         ['Bivariate'])

    def test_122_bivariate_alpha(self):
        """Perform a correlation on two data sets with alpha set to 0.02"""
        np.random.seed(self._seed)
        input_x_array = st.weibull_min.rvs(2, size=200)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertEqual(analyze(input_x_array, input_y_array,
                                 alpha=0.02,
                                 title='Alpha Test',
                                 debug=True,
                                 save_to='{}test_analyze_122'.format(self.save_path)),
                         ['Bivariate'])

    def test_123_distribution_default(self):
        """Perform a distribution analysis with default settings"""
        np.random.seed(self._seed)
        input_array = st.norm.rvs(size=200)
        self.assertEqual(analyze(input_array,
                                 debug=True,
                                 save_to='{}test_analyze_123'.format(self.save_path)),
                         ['Distribution', 'NormTest'])

    def test_124_distribution_label(self):
        """Perform a distribution analysis with label set"""
        np.random.seed(self._seed)
        input_array = st.norm.rvs(size=200)
        self.assertEqual(analyze(input_array,
                                 name='Test',
                                 title='Label Test',
                                 debug=True,
                                 save_to='{}test_analyze_124'.format(self.save_path)),
                         ['Distribution', 'NormTest'])

    def test_125_distribution_sample(self):
        """Perform a distribution analysis with sample set"""
        np.random.seed(self._seed)
        input_array = st.norm.rvs(size=200)
        self.assertEqual(analyze(input_array,
                                 sample=True,
                                 title='Sample Stats',
                                 debug=True,
                                 save_to='{}test_analyze_125'.format(self.save_path)),
                         ['Distribution', 'NormTest'])

    def test_126_distribution_cdf(self):
        """Perform a distribution analysis with cdf"""
        np.random.seed(self._seed)
        input_array = st.norm.rvs(size=200)
        self.assertEqual(analyze(input_array,
                                 cdf=True,
                                 title='CDF Test',
                                 debug=True,
                                 save_to='{}test_analyze_126'.format(self.save_path)),
                         ['Distribution', 'NormTest'])

    def test_127_distribution_fit_norm_default(self):
        """Perform a distribution analysis with normal dist KSTest"""
        np.random.seed(self._seed)
        input_array = st.norm.rvs(size=200)
        self.assertEqual(analyze(input_array,
                                 distribution='norm',
                                 fit=True,
                                 title='Norm Fit',
                                 debug=True,
                                 save_to='{}test_analyze_127'.format(self.save_path)),
                         ['Distribution', 'KSTest'])

    def test_128_distribution_fit_norm_alpha(self):
        """Perform a distribution analysis with normal dist KSTest and alpha 0.02"""
        np.random.seed(self._seed)
        input_array = st.norm.rvs(size=200)
        self.assertEqual(analyze(input_array,
                                 distribution='norm',
                                 fit=True,
                                 alpha=0.02,
                                 title='Alpha 0.02',
                                 debug=True,
                                 save_to='{}test_analyze_128'.format(self.save_path)),
                         ['Distribution', 'KSTest'])

    def test_129_distribution_categorical_default(self):
        """Perform a distribution analysis with categorical data and default settings."""
        np.random.seed(self._seed)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        self.assertListEqual(analyze(input_array,
                                     debug=True,
                                     save_to='{}test_analyze_129'.format(self.save_path)),
                             ['Frequencies'])

    def test_130_distribution_categorical_percent(self):
        """Perform a distribution analysis with categorical data and percent y-axis."""
        np.random.seed(self._seed)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        self.assertListEqual(analyze(input_array,
                                     debug=True,
                                     percent=True,
                                     save_to='{}test_analyze_130'.format(self.save_path)),
                             ['Frequencies'])

    def test_131_distribution_categorical_percent_alias(self):
        """Perform a distribution analysis with categorical data and percent y-axis using the analyse alias."""
        np.random.seed(self._seed)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        self.assertListEqual(analyse(input_array,
                                     debug=True,
                                     percent=True,
                                     save_to='{}test_analyze_131'.format(self.save_path)),
                             ['Frequencies'])

    def test_132_stacked_ttest_default(self):
        np.random.seed(self._seed)
        input_1_array = pd.DataFrame({'input': st.norm.rvs(size=2000), 'group': ['Group 1'] * 2000})
        input_2_array = pd.DataFrame({'input': st.norm.rvs(1, size=2000), 'group': ['Group 2'] * 2000})
        df = pd.concat([input_1_array, input_2_array])
        self.assertEqual(analyze(df['input'], groups=df['group'],
                                 debug=True,
                                 save_to='{}test_analyze_132'.format(self.save_path)),
                         ['Stacked Oneway', 'TTest'])

    def test_133_two_group_bivariate(self):
        """Perform a correlation with two groups."""
        np.random.seed(self._seed)
        input_1_x = st.norm.rvs(size=100)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=100)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        grp = [1] * 100 + [2] * 100
        cs_x = np.concatenate((input_1_x, input_2_x))
        cs_y = np.concatenate((input_1_y, input_2_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertEqual(analyze(input_array['a'], input_array['b'], groups=input_array['c'],
                                 debug=True,
                                 save_to='{}test_analyze_133'.format(self.save_path)),
                         ['Group Bivariate'])

    def test_134_three_group_bivariate(self):
        """Perform a correlation with three groups."""
        np.random.seed(self._seed)
        size = 100
        input_1_x = st.norm.rvs(size=size)
        input_1_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_1_x]
        input_2_x = st.norm.rvs(size=size)
        input_2_y = [(x / 2) + st.norm.rvs(0, 0.2, size=1)[0] for x in input_2_x]
        input_3_x = st.norm.rvs(size=size)
        input_3_y = np.array([(x * 1.5) + st.norm.rvs(size=1)[0] for x in input_3_x]) - 0.5
        grp = [1] * size + [2] * size + [3] * size
        cs_x = np.concatenate((input_1_x, input_2_x, input_3_x))
        cs_y = np.concatenate((input_1_y, input_2_y, input_3_y))
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertEqual(analyze(input_array['a'], input_array['b'], groups=input_array['c'],
                                 debug=True,
                                 save_to='{}test_analyze_134'.format(self.save_path)),
                         ['Group Bivariate'])

    def test_135_stacked_manwhitney_default(self):
        np.random.seed(self._seed)
        input_1_array = pd.DataFrame({'input': st.norm.rvs(size=2000), 'group': ['Group 1'] * 2000})
        input_2_array = pd.DataFrame({'input': st.weibull_min.rvs(1.2, size=2000), 'group': ['Group 2'] * 2000})
        df = pd.concat([input_1_array, input_2_array])
        self.assertEqual(analyze(df['input'], groups=df['group'],
                                 debug=True,
                                 save_to='{}test_analyze_135'.format(self.save_path)),
                         ['Stacked Oneway', 'MannWhitney'])

    def test_136_stacked_twosampleks_default(self):
        np.random.seed(self._seed)
        size = 10
        input_1_array = pd.DataFrame({'input': np.append(st.norm.rvs(0, 1, size=size), st.norm.rvs(10, 1, size=size)),
                                      'group': ['Group 1'] * size * 2})
        input_2_array = pd.DataFrame({'input': np.append(st.norm.rvs(0, 1, size=size), st.norm.rvs(10, 1, size=size)),
                                      'group': ['Group 2'] * size * 2})
        df = pd.concat([input_1_array, input_2_array])
        self.assertListEqual(analyze(df['input'], groups=df['group'],
                                     debug=True,
                                     save_to='{}test_analyze_136'.format(self.save_path)),
                             ['Stacked Oneway', 'TwoSampleKSTest'])

    def test_137_stacked_anova_default(self):
        np.random.seed(self._seed)
        size = 100
        input_1_array = pd.DataFrame({'input': st.norm.rvs(size=size), 'group': ['Group 1'] * size})
        input_2_array = pd.DataFrame({'input': st.norm.rvs(size=size), 'group': ['Group 2'] * size})
        input_3_array = pd.DataFrame({'input': st.norm.rvs(0.5, size=size), 'group': ['Group 3'] * size})
        input_4_array = pd.DataFrame({'input': st.norm.rvs(size=size), 'group': ['Group 4'] * size})
        df = pd.concat([input_1_array, input_2_array, input_3_array, input_4_array])
        self.assertEqual(analyze(df['input'], groups=df['group'],
                                 debug=True,
                                 save_to='{}test_analyze_137'.format(self.save_path)),
                         ['Stacked Oneway', 'Anova'])

    def test_138_stacked_kw_default(self):
        np.random.seed(self._seed)
        size = 100
        input_1_array = pd.DataFrame({'input': st.norm.rvs(0, 0.75, size=size), 'group': ['Group 1'] * size})
        input_2_array = pd.DataFrame({'input': st.norm.rvs(size=size), 'group': ['Group 2'] * size})
        input_3_array = pd.DataFrame({'input': st.norm.rvs(0.5, size=size), 'group': ['Group 3'] * size})
        input_4_array = pd.DataFrame({'input': st.norm.rvs(size=size), 'group': ['Group 4'] * size})
        df = pd.concat([input_1_array, input_2_array, input_3_array, input_4_array])
        self.assertEqual(analyze(df['input'], groups=df['group'],
                                 debug=True,
                                 save_to='{}test_analyze_138'.format(self.save_path)),
                         ['Stacked Oneway', 'Kruskal'])

    def test_139_stacked_two_group_mann_whitney(self):
        np.random.seed(self._seed)
        size = 42
        df = pd.DataFrame({'input': st.weibull_max.rvs(1.2, size=size),
                           'Condition': ['Group A', 'Group B'] * (size // 2)})
        self.assertEqual(analyze(df['input'], groups=df['Condition'],
                                 debug=True,
                                 save_to='{}test_analyze_139'.format(self.save_path)),
                         ['Stacked Oneway', 'MannWhitney'])

    def test_140_scatter_highlight_labels(self):
        """"""
        np.random.seed(self._seed)
        df = pd.DataFrame(np.random.randn(200, 2), columns=list('xy'))
        df['labels'] = np.random.randint(10000, 50000, size=200)
        self.assertEqual(
            analyze(
                df['x'],
                df['y'],
                labels=df['labels'],
                highlight=[39407, 11205],
                save_to='{}test_analyze_140'.format(self.save_path),
                debug=True,
            ), ['Bivariate']
        )

    def test_141_scatter_groups_one_below_min_size(self):
        np.random.seed(self._seed)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['groups'] = np.random.choice(list('ABC'), len(df)).tolist()
        df.at[24, 'groups'] = "D"
        self.assertEqual(
            analyze(
                df['x'],
                df['y'],
                df['groups'],
                debug=True,
                save_to='{}test_analyze_141'.format(self.save_path)
            ),
            ['Group Bivariate']
        )

    def test_142_stacked_oneway_missing_groups(self):
        np.random.seed(self._seed)
        size = 100
        input_1_array = pd.DataFrame({'input': st.norm.rvs(0, 0.75, size=size), 'group': ['Group 1'] * size})
        input_2_array = pd.DataFrame({'input': [np.nan] * size, 'group': ['Group 2'] * size})
        input_3_array = pd.DataFrame({'input': st.norm.rvs(0.5, size=size), 'group': ['Group 3'] * size})
        input_4_array = pd.DataFrame({'input': [np.nan] * size, 'group': ['Group 4'] * size})
        df = pd.concat([input_1_array, input_2_array, input_3_array, input_4_array])
        self.assertEqual(analyze(df['input'], groups=df['group'],
                                 debug=True,
                                 save_to='{}test_analyze_142'.format(self.save_path)),
                         ['Stacked Oneway', 'TTest'])

    def test_143_categorical_ordered(self):
        input_array = ['one', 'two', 'one', 'three', 'one', 'three', 'three', 'one']
        self.assertEqual(analyze(
            input_array,
            order=['one', 'two', 'three'],
            debug=True,
            save_to='{}test_analyze_143'.format(self.save_path)
        ), ['Frequencies'])

    def test_144_categorical_no_labels(self):
        input_array = ['one', 'two', 'one', 'three', 'one', 'three', 'three', 'one']
        self.assertEqual(analyze(
            input_array,
            labels=False,
            debug=True,
            save_to='{}test_analyze_144'.format(self.save_path)
        ), ['Frequencies'])

    def test_145_categorical_with_grid(self):
        input_array = ['one', 'two', 'one', 'three', 'one', 'three', 'three', 'one']
        self.assertEqual(analyze(
            input_array,
            grid=True,
            debug=True,
            save_to='{}test_analyze_145'.format(self.save_path)
        ), ['Frequencies'])


if __name__ == '__main__':
    unittest.main()
