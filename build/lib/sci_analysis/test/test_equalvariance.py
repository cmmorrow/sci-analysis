import unittest
import numpy as np
import scipy.stats as st

from ..analysis import EqualVariance
from ..analysis.exc import MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    def test_450_EqualVariance_Bartlett_matched(self):
        """Test the EqualVariance class for normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        exp = EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False)
        output = """

Bartlett Test
-------------

alpha   =  0.0500
T value =  0.2264
p value =  0.8930

H0: Variances are equal
"""
        self.assertGreater(exp.p_value, a, "FAIL: Equal variance Bartlett Type I error")
        self.assertEqual(exp.test_type, 'Bartlett')
        self.assertAlmostEqual(exp.statistic, 0.2264, delta=0.0001)
        self.assertAlmostEqual(exp.t_value, 0.2264, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.893, delta=0.001)
        self.assertEqual(str(exp), output)

    def test_452_EqualVariance_Bartlett_unmatched(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        exp = EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=True)
        output = """

Bartlett Test
-------------

alpha   =  0.0500
T value =  43.0402
p value =  0.0000

HA: Variances are not equal
"""
        self.assertLess(exp.p_value, a, "FAIL: Equal variance bartlett Type II error")
        self.assertEqual(exp.test_type, 'Bartlett')
        self.assertAlmostEqual(exp.statistic, 43.0402, delta=0.0001)
        self.assertAlmostEqual(exp.t_value, 43.0402, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_456_EqualVariance_Bartlett_unmatched_w_value(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertRaises(KeyError, lambda: EqualVariance(x_input_array, y_input_array, z_input_array,
                                                          alpha=a,
                                                          display=False).w_value)

    def test_457_EqualVariance_Bartlett_single_argument(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(NoDataError, lambda: EqualVariance(x_input_array, alpha=a, display=False).p_value)

    def test_458_EqualVariance_Levene_matched(self):
        """Test the EqualVariance class for non-normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        z_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        a = 0.05
        exp = EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False)
        output = """

Levene Test
-----------

alpha   =  0.0500
W value =  1.7545
p value =  0.1748

H0: Variances are equal
"""
        self.assertGreater(exp.p_value, a, "FAIL: Unequal variance levene Type I error")
        self.assertEqual(exp.test_type, 'Levene')
        self.assertAlmostEqual(exp.statistic, 1.7545, delta=0.0001)
        self.assertAlmostEqual(exp.w_value, 1.7545, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.1748, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_460_EqualVariance_Levene_unmatched(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        exp = EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=True)
        output = """

Levene Test
-----------

alpha   =  0.0500
W value =  11.2166
p value =  0.0000

HA: Variances are not equal
"""
        self.assertLess(exp.p_value, a, "FAIL: Unequal variance levene Type II error")
        self.assertEqual(exp.test_type, 'Levene')
        self.assertAlmostEqual(exp.statistic, 11.2166, delta=0.0001)
        self.assertAlmostEqual(exp.w_value, 11.2166, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_463_EqualVariance_Levene_single_argument(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(NoDataError, lambda: EqualVariance(x_input_array, alpha=a, display=False).p_value)

    def test_464_EqualVariance_Levene_unmatched_t_value(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertRaises(KeyError, lambda: EqualVariance(x_input_array,
                                                          y_input_array,
                                                          z_input_array,
                                                          alpha=a,
                                                          display=False).t_value)

    def test_465_EqualVariance_Bartlett_matched_just_above_min_size(self):
        """Test the EqualVariance class for normally distributed matched variances just above min size"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=3)
        y_input_array = st.norm.rvs(*y_parms, size=3)
        z_input_array = st.norm.rvs(*z_parms, size=3)
        a = 0.05
        exp = EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False)
        output = """

Bartlett Test
-------------

alpha   =  0.0500
T value =  0.0785
p value =  0.9615

H0: Variances are equal
"""
        self.assertGreater(exp.p_value, a, "FAIL: Equal variance Bartlett just above min size")
        self.assertEqual(exp.test_type, 'Bartlett')
        self.assertAlmostEqual(exp.statistic, 0.0785, delta=0.0001)
        self.assertAlmostEqual(exp.t_value, 0.0785, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.9615, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_466_EqualVariance_Bartlett_matched_at_min_size(self):
        """Test the EqualVariance class for normally distributed matched variances at min size"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=2)
        y_input_array = st.norm.rvs(*y_parms, size=9)
        z_input_array = st.norm.rvs(*z_parms, size=47)
        a = 0.05
        self.assertTrue(MinimumSizeError, lambda: EqualVariance(x_input_array, y_input_array, z_input_array,
                                                                alpha=a,
                                                                display=False).p_value)

    def test_467_EqualVariance_Bartlett_matched_single_empty_vector(self):
        """Test the EqualVariance class for normally distributed matched variances single empty vector"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = ["one", "two", "three", "four", "five"]
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        exp = EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False)
        output = """

Bartlett Test
-------------

alpha   =  0.0500
T value =  0.0374
p value =  0.8466

H0: Variances are equal
"""
        self.assertGreater(exp.p_value, a)
        self.assertEqual(str(exp), output)

    def test_466_EqualVariance_Bartlett_all_empty_vectors(self):
        """Test the EqualVariance class for normally distributed matched variances with all empty vectors"""
        np.random.seed(987654321)
        x_input_array = [np.nan, np.nan, np.nan, "four", np.nan]
        y_input_array = ["one", "two", "three", "four", "five"]
        a = 0.05
        self.assertTrue(NoDataError, lambda: EqualVariance(x_input_array, y_input_array,
                                                           alpha=a,
                                                           display=False).p_value)


if __name__ == '__main__':
    unittest.main()
