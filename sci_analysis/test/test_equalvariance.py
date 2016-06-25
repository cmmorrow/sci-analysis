import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import EqualVariance


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
        self.assertGreater(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value,
                           a,
                           "FAIL: Equal variance Bartlett Type I error")

    def test_451_EqualVariance_Bartlett_matched_test_type(self):
        """Test the EqualVariance class for normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Bartlett",
                         "FAIL: Equal variance Bartlett test type")

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
        self.assertLess(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value, a,
                        "FAIL: Equal variance bartlett Type II error")

    def test_453_EqualVariance_Bartlett_unmatched_test_type(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Bartlett",
                         "FAIL: Equal variance bartlett test type")

    def test_454_EqualVariance_Bartlett_unmatched_statistic(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(EqualVariance(x_input_array, y_input_array, z_input_array,
                                             alpha=a,
                                             display=False).statistic,
                               43.0402,
                               delta=0.0001,
                               msg="FAIL: Equal variance bartlett statistic")

    def test_455_EqualVariance_Bartlett_unmatched_t_value(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(EqualVariance(x_input_array, y_input_array, z_input_array,
                                             alpha=a,
                                             display=False).t_value,
                               43.0402,
                               delta=0.0001,
                               msg="FAIL: Equal variance bartlett t value")

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

    # TODO: Update this to use a specific exception in the future
    def test_457_EqualVariance_Bartlett_single_argument(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(TypeError, lambda: EqualVariance(x_input_array, alpha=a, display=False).p_value)

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
        self.assertGreater(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value,
                           a,
                           "FAIL: Unequal variance levene Type I error")

    def test_459_EqualVariance_Levene_matched_test_type(self):
        """Test the EqualVariance class for non-normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        z_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        a = 0.05
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Levene",
                         "FAIL: Unequal variance levene test type")

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
        self.assertLess(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value, a,
                        "FAIL: Unequal variance levene Type II error")

    def test_461_EqualVariance_Levene_unmatched_test_type(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Levene",
                         "FAIL: Unequal variance levene test type")

    def test_462_EqualVariance_Levene_unmatched_statistic(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertAlmostEqual(EqualVariance(x_input_array,
                                             y_input_array,
                                             z_input_array,
                                             alpha=a,
                                             display=False).statistic,
                               11.2166,
                               delta=0.0001,
                               msg="FAIL: Unequal variance levene statistic")

    def test_463_EqualVariance_Levene_unmatched_w_value(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertAlmostEqual(EqualVariance(x_input_array,
                                             y_input_array,
                                             z_input_array,
                                             alpha=a,
                                             display=False).w_value,
                               11.2166,
                               delta=0.0001,
                               msg="FAIL: Unequal variance levene w value")

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


if __name__ == '__main__':
    unittest.main()
