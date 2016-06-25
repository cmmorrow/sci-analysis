import unittest
import scipy.stats as st
import numpy as np

from ..analysis.analysis import TTest


class MyTestCase(unittest.TestCase):
    # Test TTest

    def test_200_TTest_single_matched(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        alpha = 0.05
        self.assertTrue(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).p_value > alpha,
                        "FAIL: TTest single type I error")

    def test_201_TTest_single_matched_test_type(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).test_type, '1_sample',
                         "FAIL: TTest incorrect test type")

    def test_202_TTest_single_matched_mu(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).mu, y_val,
                         "FAIL: TTest incorrect mu")

    def test_203_TTest_single_matched_t_value(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertTrue(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).t_value,
                        "FAIL: TTest t value not set")

    def test_204_TTest_single_matched_statistic(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertTrue(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).statistic,
                        "FAIL: TTest statistic not set")

    def test_205_TTest_single_unmatched(self):
        """Test the TTest against a given unmatched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 5.0
        alpha = 0.05
        self.assertFalse(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).p_value > alpha,
                         "FAIL: TTest single type II error")

    def test_206_TTest_equal_variance_matched(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        alpha = 0.05
        self.assertGreater(TTest(st.norm.rvs(*x_parms, size=100),
                                 st.norm.rvs(*y_parms, size=100), display=False).p_value,
                           alpha, "FAIL: TTest equal variance matched Type I error")

    def test_207_TTest_equal_variance_matched_test_type(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100),
                               st.norm.rvs(*y_parms, size=100), display=False).test_type, 't_test',
                         "FAIL: TTest incorrect test type")

    def test_208_TTest_equal_variance_matched_t_value(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        self.assertAlmostEqual(TTest(st.norm.rvs(*x_parms, size=100),
                                     st.norm.rvs(*y_parms, size=100), display=False).t_value, -0.2592,
                               msg="FAIL: TTest equal variance matched Type I error", delta=0.0001)

    def test_209_TTest_equal_variance_unmatched(self):
        """Test the TTest with two samples with equal variance and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 0.75]
        alpha = 0.05
        self.assertLess(TTest(st.norm.rvs(*x_parms, size=100),
                              st.norm.rvs(*y_parms, size=100), display=False).p_value, alpha,
                        "FAIL: TTest equal variance unmatched Type II error")

    def test_210_TTest_unequal_variance_matched(self):
        """Test the TTest with two samples with different variances and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 1.35]
        alpha = 0.05
        self.assertGreater(TTest(st.norm.rvs(*x_parms, size=100),
                                 st.norm.rvs(*y_parms, size=100), display=False).p_value, alpha,
                           "FAIL: TTest different variance matched Type I error")

    def test_211_TTest_unequal_variance_unmatched(self):
        """Test the TTest with two samples with different variances and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 1.12]
        alpha = 0.05
        self.assertLess(TTest(st.norm.rvs(*x_parms, size=100),
                              st.norm.rvs(*y_parms, size=100), display=False).p_value, alpha,
                        "FAIL: TTest different variance unmatched Type II error")

    def test_212_TTest_unequal_variance_unmatched_test_type(self):
        """Test the TTest with two samples with different variances and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 1.12]
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100),
                               st.norm.rvs(*y_parms, size=100), display=False).test_type, 'welch_t',
                         "FAIL: TTest incorrect test type")

    def test_213_TTest_equal_variance_matched_too_many_args(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        alpha = 0.05
        self.assertGreater(TTest(st.norm.rvs(*x_parms, size=100),
                                 st.norm.rvs(*y_parms, size=100),
                                 st.norm.rvs(*y_parms, size=100),
                                 display=False).p_value,
                           alpha, "FAIL: TTest equal variance matched ok with too many args")

    def test_214_TTest_equal_variance_matched_min_size(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        alpha = 0.05
        self.assertGreater(TTest(st.norm.rvs(*x_parms, size=4),
                                 st.norm.rvs(*y_parms, size=4), display=False).p_value,
                           alpha, "FAIL: TTest equal variance matched ok with too many args")


if __name__ == '__main__':
    unittest.main()
