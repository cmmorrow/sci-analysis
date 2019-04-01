import unittest
import numpy as np
import pandas as pd
import scipy.stats as st

from ..analysis import GroupLinearRegression
from ..analysis.exc import MinimumSizeError, NoDataError
from ..data import UnequalVectorLengthError, Vector


class MyTestCase(unittest.TestCase):

    def test_linregress_four_groups(self):
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

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
100           -0.0056        0.0478        0.0000        0.1030        0.9567       1             
100            0.0570       -0.1671        0.0037        0.0950        0.5497       2             
100           -0.2521        0.1637        0.0506        0.1103        0.0244       3             
100            0.9635        0.1043        0.8181        0.0459        0.0000       4             """
        exp = GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertTupleEqual(exp.counts, ('100', '100', '100', '100'))
        self.assertAlmostEqual(exp.slope[0], -0.005613130406764816)
        self.assertAlmostEqual(exp.slope[1], 0.0570354136308546)
        self.assertAlmostEqual(exp.slope[2], -0.2521496921022714)
        self.assertAlmostEqual(exp.slope[3], 0.9634599098599703)
        self.assertAlmostEqual(exp.intercept[0], 0.04775111565537506)
        self.assertAlmostEqual(exp.intercept[1], -0.1670688836199169)
        self.assertAlmostEqual(exp.intercept[2], 0.1637132078993005)
        self.assertAlmostEqual(exp.intercept[3], 0.10434448563066669)
        self.assertAlmostEqual(exp.r_squared[0], 3.030239852495909e-05)
        self.assertAlmostEqual(exp.r_squared[1], 0.00366271257512563)
        self.assertAlmostEqual(exp.r_squared[2], 0.05062765121282169)
        self.assertAlmostEqual(exp.r_squared[3], 0.8180520671815105)
        self.assertAlmostEqual(exp.statistic[0], 3.030239852495909e-05)
        self.assertAlmostEqual(exp.statistic[1], 0.00366271257512563)
        self.assertAlmostEqual(exp.statistic[2], 0.05062765121282169)
        self.assertAlmostEqual(exp.statistic[3], 0.8180520671815105)
        self.assertAlmostEqual(exp.r_value[0], -0.005504761441239674)
        self.assertAlmostEqual(exp.r_value[1], 0.06052034843856759)
        self.assertAlmostEqual(exp.r_value[2], -0.2250058915069152)
        self.assertAlmostEqual(exp.r_value[3], 0.9044623083255103)
        self.assertAlmostEqual(exp.std_err[0], 0.1030023210648352)
        self.assertAlmostEqual(exp.std_err[1], 0.09502400478678666)
        self.assertAlmostEqual(exp.std_err[2], 0.11029855015697929)
        self.assertAlmostEqual(exp.std_err[3], 0.04589905033402483)
        self.assertAlmostEqual(exp.p_value[0], 0.956651586890106)
        self.assertAlmostEqual(exp.p_value[1], 0.5497443545114141)
        self.assertAlmostEqual(exp.p_value[2], 0.024403659194742487)
        self.assertAlmostEqual(exp.p_value[3], 4.844813765580163e-38)
        self.assertEqual(str(exp), output)

    def test_linregress_four_groups_string(self):
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_2 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_3 = st.norm.rvs(size=100), st.norm.rvs(size=100)
        input_4_x = st.norm.rvs(size=100)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = ['a'] * 100 + ['b'] * 100 + ['c'] * 100 + ['d'] * 100
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
100           -0.0056        0.0478        0.0000        0.1030        0.9567       a             
100            0.0570       -0.1671        0.0037        0.0950        0.5497       b             
100           -0.2521        0.1637        0.0506        0.1103        0.0244       c             
100            0.9635        0.1043        0.8181        0.0459        0.0000       d             """
        exp = GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertTupleEqual(exp.counts, ('100', '100', '100', '100'))
        self.assertAlmostEqual(exp.slope[0], -0.005613130406764816)
        self.assertAlmostEqual(exp.slope[1], 0.0570354136308546)
        self.assertAlmostEqual(exp.slope[2], -0.2521496921022714)
        self.assertAlmostEqual(exp.slope[3], 0.9634599098599703)
        self.assertAlmostEqual(exp.intercept[0], 0.04775111565537506)
        self.assertAlmostEqual(exp.intercept[1], -0.1670688836199169)
        self.assertAlmostEqual(exp.intercept[2], 0.1637132078993005)
        self.assertAlmostEqual(exp.intercept[3], 0.10434448563066669)
        self.assertAlmostEqual(exp.r_squared[0], 3.030239852495909e-05)
        self.assertAlmostEqual(exp.r_squared[1], 0.00366271257512563)
        self.assertAlmostEqual(exp.r_squared[2], 0.05062765121282169)
        self.assertAlmostEqual(exp.r_squared[3], 0.8180520671815105)
        self.assertAlmostEqual(exp.statistic[0], 3.030239852495909e-05)
        self.assertAlmostEqual(exp.statistic[1], 0.00366271257512563)
        self.assertAlmostEqual(exp.statistic[2], 0.05062765121282169)
        self.assertAlmostEqual(exp.statistic[3], 0.8180520671815105)
        self.assertAlmostEqual(exp.r_value[0], -0.005504761441239674)
        self.assertAlmostEqual(exp.r_value[1], 0.06052034843856759)
        self.assertAlmostEqual(exp.r_value[2], -0.2250058915069152)
        self.assertAlmostEqual(exp.r_value[3], 0.9044623083255103)
        self.assertAlmostEqual(exp.std_err[0], 0.1030023210648352)
        self.assertAlmostEqual(exp.std_err[1], 0.09502400478678666)
        self.assertAlmostEqual(exp.std_err[2], 0.11029855015697929)
        self.assertAlmostEqual(exp.std_err[3], 0.04589905033402483)
        self.assertAlmostEqual(exp.p_value[0], 0.956651586890106)
        self.assertAlmostEqual(exp.p_value[1], 0.5497443545114141)
        self.assertAlmostEqual(exp.p_value[2], 0.024403659194742487)
        self.assertAlmostEqual(exp.p_value[3], 4.844813765580163e-38)
        self.assertEqual(str(exp), output)

    def test_no_data(self):
        """Test the case where there's no data."""
        self.assertRaises(NoDataError, lambda: GroupLinearRegression([], []))

    def test_at_minimum_size(self):
        """Test to make sure the case where the length of data is just above the minimum size."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        input_2 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        input_3 = st.norm.rvs(size=2), st.norm.rvs(size=2)
        input_4_x = st.norm.rvs(size=2)
        input_4_y = [x + st.norm.rvs(0, 0.5, size=1)[0] for x in input_4_x]
        input_4 = input_4_x, input_4_y
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 2 + [2] * 2 + [3] * 2 + [4] * 2
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
2             -1.0763        1.2343        1.0000        0.0000        0.0000       1             
2              2.0268        0.6799        1.0000        0.0000        0.0000       2             
2              1.8891       -2.4800        1.0000        0.0000        0.0000       3             
2              0.1931       -0.2963        1.0000        0.0000        0.0000       4             """
        exp = GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertEqual(str(exp), output)

    def test_all_below_minimum_size(self):
        """Test the case where all the supplied data is less than the minimum size."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_2 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_3 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_4 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1, 2, 3, 4]
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertRaises(
            NoDataError,
            lambda: GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c'])
        )

    def test_below_minimum_size(self):
        """Test the case where a group is less than the minimum size."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=10), st.norm.rvs(size=10)
        input_2 = st.norm.rvs(size=10), st.norm.rvs(size=10)
        input_3 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_4 = st.norm.rvs(size=10), st.norm.rvs(size=10)
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1] * 10 + [2] * 10 + [3] + [4] * 10
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        output = """

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
10             0.4268       -0.2032        0.2877        0.2374        0.1100       1             
10             0.1214       -0.6475        0.0393        0.2123        0.5832       2             
10             0.2367        0.2525        0.1131        0.2343        0.3419       4             """
        exp = GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c'])
        self.assertEqual(output, str(exp))

    def test_vector_no_data(self):
        """Test the case where there's no data with a vector as input."""
        self.assertRaises(NoDataError, lambda: GroupLinearRegression(Vector([], other=[])))

    def test_no_ydata(self):
        """Test the case where the ydata argument is None."""
        self.assertRaises(AttributeError, lambda: GroupLinearRegression([1, 2, 3, 4]))

    def test_unequal_pair_lengths(self):
        """Test the case where the supplied pairs are unequal."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=100), st.norm.rvs(size=96)
        self.assertRaises(UnequalVectorLengthError, lambda: GroupLinearRegression(input_1[0], input_1[1]))

    def test_linregress_one_group(self):
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100), st.norm.rvs(size=100)
        output = """

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
100           -0.0056        0.0478        0.0000        0.1030        0.9567       1             """
        exp = GroupLinearRegression(input_array[0], input_array[1], display=False)
        self.assertEqual(str(exp), output)

    def test_linregress_vector(self):
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

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
100           -0.0056        0.0478        0.0000        0.1030        0.9567       1             
100            0.0570       -0.1671        0.0037        0.0950        0.5497       2             
100           -0.2521        0.1637        0.0506        0.1103        0.0244       3             
100            0.9635        0.1043        0.8181        0.0459        0.0000       4             """
        exp = GroupLinearRegression(input_array, display=False)
        self.assertTupleEqual(exp.counts, ('100', '100', '100', '100'))
        self.assertAlmostEqual(exp.slope[0], -0.005613130406764816)
        self.assertAlmostEqual(exp.slope[1], 0.0570354136308546)
        self.assertAlmostEqual(exp.slope[2], -0.2521496921022714)
        self.assertAlmostEqual(exp.slope[3], 0.9634599098599703)
        self.assertAlmostEqual(exp.intercept[0], 0.04775111565537506)
        self.assertAlmostEqual(exp.intercept[1], -0.1670688836199169)
        self.assertAlmostEqual(exp.intercept[2], 0.1637132078993005)
        self.assertAlmostEqual(exp.intercept[3], 0.10434448563066669)
        self.assertAlmostEqual(exp.r_squared[0], 3.030239852495909e-05)
        self.assertAlmostEqual(exp.r_squared[1], 0.00366271257512563)
        self.assertAlmostEqual(exp.r_squared[2], 0.05062765121282169)
        self.assertAlmostEqual(exp.r_squared[3], 0.8180520671815105)
        self.assertAlmostEqual(exp.statistic[0], 3.030239852495909e-05)
        self.assertAlmostEqual(exp.statistic[1], 0.00366271257512563)
        self.assertAlmostEqual(exp.statistic[2], 0.05062765121282169)
        self.assertAlmostEqual(exp.statistic[3], 0.8180520671815105)
        self.assertAlmostEqual(exp.r_value[0], -0.005504761441239674)
        self.assertAlmostEqual(exp.r_value[1], 0.06052034843856759)
        self.assertAlmostEqual(exp.r_value[2], -0.2250058915069152)
        self.assertAlmostEqual(exp.r_value[3], 0.9044623083255103)
        self.assertAlmostEqual(exp.std_err[0], 0.1030023210648352)
        self.assertAlmostEqual(exp.std_err[1], 0.09502400478678666)
        self.assertAlmostEqual(exp.std_err[2], 0.11029855015697929)
        self.assertAlmostEqual(exp.std_err[3], 0.04589905033402483)
        self.assertAlmostEqual(exp.p_value[0], 0.956651586890106)
        self.assertAlmostEqual(exp.p_value[1], 0.5497443545114141)
        self.assertAlmostEqual(exp.p_value[2], 0.024403659194742487)
        self.assertAlmostEqual(exp.p_value[3], 4.844813765580163e-38)
        self.assertEqual(str(exp), output)

    def test_linregress_missing_data(self):
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

Linear Regression
-----------------

n             Slope         Intercept     r^2           Std Err       p value       Group         
--------------------------------------------------------------------------------------------------
99            -0.0115        0.0340        0.0001        0.1028        0.9114       1             
99             0.0281       -0.1462        0.0009        0.0950        0.7681       2             
99            -0.2546        0.1653        0.0495        0.1133        0.0269       3             
99             0.9635        0.1043        0.8178        0.0462        0.0000       4             """
        exp = GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c'], display=False)
        self.assertEqual(str(exp), output)


if __name__ == '__main__':
    unittest.main()
