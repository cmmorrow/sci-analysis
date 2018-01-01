import unittest
import numpy as np
import pandas as pd
import scipy.stats as st

from ..analysis import GroupCorrelation
from ..analysis.exc import MinimumSizeError, NoDataError
from ..data import UnequalVectorLengthError, Vector


class MyTestCase(unittest.TestCase):

    def test_pearson_correlation_four_groups(self):
        """Test the output of a pearson correlation with three groups."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_2 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_3 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_4_x = st.norm.rvs(size=100)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Pearson Correlation Coefficient
-------------------------------

n             r value       p value       Group         
--------------------------------------------------------
100           -0.0055        0.9567       1             
100            0.0605        0.5497       2             
100           -0.2250        0.0244       3             
100            0.9045        0.0000       4             """
        exp = GroupCorrelation(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertTupleEqual(('100', '100', '100', '100'), exp.counts)
        self.assertTupleEqual((-0.005504761441239719, 0.06052034843856759, -0.225005891506915, 0.9044623083255101),
                              exp.r_value)
        self.assertTupleEqual((-0.005504761441239719, 0.06052034843856759, -0.225005891506915, 0.9044623083255101),
                              exp.statistic)
        self.assertTupleEqual((0.9566515868901755, 0.5497443545114141, 0.02440365919474257, 4.844813765580646e-38),
                              exp.p_value)
        self.assertEqual(str(exp), output)

    def test_pearson_correlation_four_string_groups(self):
        """Test the output of a pearson correlation with three groups."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_2 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_3 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_4_x = st.norm.rvs(size=100)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = ['b'] * 100 + ['a'] * 100 + ['d'] * 100 + ['c'] * 100
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Pearson Correlation Coefficient
-------------------------------

n             r value       p value       Group         
--------------------------------------------------------
100            0.0605        0.5497       a             
100           -0.0055        0.9567       b             
100            0.9045        0.0000       c             
100           -0.2250        0.0244       d             """
        exp = GroupCorrelation(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertTupleEqual(('100', '100', '100', '100'), exp.counts)
        self.assertTupleEqual((0.06052034843856759, -0.005504761441239719, 0.9044623083255101, -0.225005891506915),
                              exp.r_value)
        self.assertTupleEqual((0.06052034843856759, -0.005504761441239719, 0.9044623083255101, -0.225005891506915),
                              exp.statistic)
        self.assertTupleEqual((0.5497443545114141, 0.9566515868901755, 4.844813765580646e-38, 0.02440365919474257),
                              exp.p_value)
        self.assertEqual(str(exp), output)

    def test_pearson_correlation_one_groups(self):
        """Test the output of a pearson correlation with one groups."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        output = """

Pearson Correlation Coefficient
-------------------------------

n             r value       p value       Group         
--------------------------------------------------------
100           -0.0055        0.9567       1             """
        exp = GroupCorrelation(input_1[0], input_1[1], display=False)
        self.assertEqual(str(exp), output)

    def test_spearman_correlation_four_groups(self):
        """Test the output of a pearson correlation with three groups."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_2 = st.weibull_min.rvs(1.7, size=100), st.norm.rvs(size=100)
        input_3 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_4_x = st.norm.rvs(size=100)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Spearman Correlation Coefficient
--------------------------------

n             r value       p value       Group         
--------------------------------------------------------
100            0.0079        0.9376       1             
100            0.0140        0.8898       2             
100           -0.1227        0.2241       3             
100            0.9006        0.0000       4             """
        exp = GroupCorrelation(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertTupleEqual(('100', '100', '100', '100'), exp.counts)
        self.assertTupleEqual((0.007932793279327931, 0.014029402940294028, -0.12266426642664265, 0.9005940594059406),
                              exp.r_value)
        self.assertTupleEqual((0.007932793279327931, 0.014029402940294028, -0.12266426642664265, 0.9005940594059406),
                              exp.statistic)
        self.assertTupleEqual((0.9375641178035645, 0.8898160391011217, 0.22405419866382636, 3.0794115586718083e-37),
                              exp.p_value)
        self.assertEqual(str(exp), output)

    def test_pearson_correlation_with_missing_data(self):
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_2 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_3 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_4_x = st.norm.rvs(size=100)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        input_array['a'][24] = np.nan
        input_array['a'][256] = np.nan
        input_array['b'][373] = np.nan
        input_array['b'][24] = np.nan
        input_array['b'][128] = np.nan
        output = """

Pearson Correlation Coefficient
-------------------------------

n             r value       p value       Group         
--------------------------------------------------------
99            -0.0113        0.9114       1             
99             0.0300        0.7681       2             
99            -0.2224        0.0269       3             
99             0.9043        0.0000       4             """
        exp = GroupCorrelation(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertTupleEqual(('99', '99', '99', '99'), exp.counts)
        self.assertEqual(str(exp), output)

    def test_no_data(self):
        """Test the case where there's no data."""
        self.assertRaises(NoDataError, lambda: GroupCorrelation([], []))

    def test_at_minimum_size(self):
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=3), st.norm.rvs(size=3)
        input_2 = st.norm.rvs(size=3), st.norm.rvs(size=3)
        input_3 = st.norm.rvs(size=3), st.norm.rvs(size=3)
        input_4 = st.norm.rvs(size=3), st.norm.rvs(size=3)
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Pearson Correlation Coefficient
-------------------------------

n             r value       p value       Group         
--------------------------------------------------------
3              0.9757        0.1407       1             
3             -0.8602        0.3406       2             
3              0.9530        0.1959       3             
3             -0.9981        0.0398       4             """
        exp = GroupCorrelation(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertEqual(str(exp), output)

    def test_minimum_size_error(self):
        """Test the case where the supplied data is less than the minimum size."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        input_2 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        input_3 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        input_4 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1, 1, 2, 2, 3, 3, 4, 4]
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertRaises(MinimumSizeError,
                          lambda: GroupCorrelation(input_array['a'], input_array['b'], groups=input_array['c']))

    def test_pearson_correlation_vector(self):
        """Test the case with vector input."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_2 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_3 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_4_x = st.norm.rvs(size=100)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100
        input_array = Vector(cs_x, other=cs_y, groups=grp)
        output = """

Pearson Correlation Coefficient
-------------------------------

n             r value       p value       Group         
--------------------------------------------------------
100           -0.0055        0.9567       1             
100            0.0605        0.5497       2             
100           -0.2250        0.0244       3             
100            0.9045        0.0000       4             """
        exp = GroupCorrelation(input_array, display=False)
        self.assertTupleEqual(('100', '100', '100', '100'), exp.counts)
        self.assertTupleEqual((-0.005504761441239719, 0.06052034843856759, -0.225005891506915, 0.9044623083255101),
                              exp.r_value)
        self.assertTupleEqual((-0.005504761441239719, 0.06052034843856759, -0.225005891506915, 0.9044623083255101),
                              exp.statistic)
        self.assertTupleEqual((0.9566515868901755, 0.5497443545114141, 0.02440365919474257, 4.844813765580646e-38),
                              exp.p_value)
        self.assertEqual(str(exp), output)

    def test_vector_no_data(self):
        """Test the case where there's no data with a vector as input."""
        self.assertRaises(NoDataError, lambda: GroupCorrelation(Vector([], other=[])))

    def test_no_ydata(self):
        """Test the case where the ydata argument is None."""
        self.assertRaises(AttributeError, lambda: GroupCorrelation([1, 2, 3, 4]))

    def test_unequal_pair_lengths(self):
        """Test the case where the supplied pairs are unequal."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=96)
        self.assertRaises(UnequalVectorLengthError, lambda: GroupCorrelation(input_1[0], input_1[1]))


if __name__ == '__main__':
    unittest.main()
