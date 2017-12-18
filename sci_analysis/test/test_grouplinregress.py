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
        self.assertTupleEqual(exp.slope,
                              (-0.005613130406764816, 0.0570354136308546, -0.2521496921022714, 0.9634599098599703))
        self.assertTupleEqual(exp.intercept,
                              (0.04775111565537506, -0.1670688836199169, 0.1637132078993005, 0.10434448563066669))
        self.assertTupleEqual(exp.r_squared,
                              (3.030239852495909e-05, 0.00366271257512563, 0.05062765121282169, 0.8180520671815105))
        self.assertTupleEqual(exp.r_value,
                              (-0.005504761441239674, 0.06052034843856759, -0.2250058915069152, 0.9044623083255103))
        self.assertTupleEqual(exp.std_err,
                              (0.1030023210648352, 0.09502400478678666, 0.11029855015697929, 0.04589905033402483))
        self.assertTupleEqual(exp.p_value,
                              (0.956651586890106, 0.5497443545114141, 0.024403659194742487, 4.844813765580163e-38))
        # print(exp.counts)
        # print(exp.slope)
        # print(exp.intercept)
        # print(exp.r_squared)
        # print(exp.r_value)
        # print(exp.std_err)
        # print(exp.p_value)
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
        self.assertTupleEqual(exp.slope,
                              (-0.005613130406764816, 0.0570354136308546, -0.2521496921022714, 0.9634599098599703))
        self.assertTupleEqual(exp.intercept,
                              (0.04775111565537506, -0.1670688836199169, 0.1637132078993005, 0.10434448563066669))
        self.assertTupleEqual(exp.r_squared,
                              (3.030239852495909e-05, 0.00366271257512563, 0.05062765121282169, 0.8180520671815105))
        self.assertTupleEqual(exp.r_value,
                              (-0.005504761441239674, 0.06052034843856759, -0.2250058915069152, 0.9044623083255103))
        self.assertTupleEqual(exp.std_err,
                              (0.1030023210648352, 0.09502400478678666, 0.11029855015697929, 0.04589905033402483))
        self.assertTupleEqual(exp.p_value,
                              (0.956651586890106, 0.5497443545114141, 0.024403659194742487, 4.844813765580163e-38))
        # print(exp.counts)
        # print(exp.slope)
        # print(exp.intercept)
        # print(exp.r_squared)
        # print(exp.r_value)
        # print(exp.std_err)
        # print(exp.p_value)
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

    def test_minimum_size_error(self):
        """Test the case where the supplied data is less than the minimum size."""
        np.random.seed(987654321)
        input_1 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_2 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_3 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        input_4 = st.norm.rvs(size=1), st.norm.rvs(size=1)
        cs_x = np.concatenate((input_1[0], input_2[0], input_3[0], input_4[0]))
        cs_y = np.concatenate((input_1[1], input_2[1], input_3[1], input_4[1]))
        grp = [1, 2, 3, 4]
        input_array = pd.DataFrame({'a': cs_x, 'b': cs_y, 'c': grp})
        self.assertRaises(MinimumSizeError,
                          lambda: GroupLinearRegression(input_array['a'], input_array['b'], groups=input_array['c']))

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
        self.assertTupleEqual(exp.slope,
                              (-0.005613130406764816, 0.0570354136308546, -0.2521496921022714, 0.9634599098599703))
        self.assertTupleEqual(exp.intercept,
                              (0.04775111565537506, -0.1670688836199169, 0.1637132078993005, 0.10434448563066669))
        self.assertTupleEqual(exp.r_squared,
                              (3.030239852495909e-05, 0.00366271257512563, 0.05062765121282169, 0.8180520671815105))
        self.assertTupleEqual(exp.r_value,
                              (-0.005504761441239674, 0.06052034843856759, -0.2250058915069152, 0.9044623083255103))
        self.assertTupleEqual(exp.std_err,
                              (0.1030023210648352, 0.09502400478678666, 0.11029855015697929, 0.04589905033402483))
        self.assertTupleEqual(exp.p_value,
                              (0.956651586890106, 0.5497443545114141, 0.024403659194742487, 4.844813765580163e-38))
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
