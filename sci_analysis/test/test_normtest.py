import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import NormTest


class MyTestCase(unittest.TestCase):
    def test_300_Norm_test_single(self):
        """Test the normal distribution check"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertGreater(NormTest(st.norm.rvs(*parms, size=100), display=False, alpha=alpha).p_value, alpha,
                           "FAIL: Normal test Type I error")

    def test_301_Norm_test_single_fail(self):
        """Test the normal distribution check fails for a different distribution"""
        np.random.seed(987654321)
        parms = [1.7]
        alpha = 0.05
        self.assertLess(NormTest(st.weibull_min.rvs(*parms, size=100), alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Normal test Type II error")

    def test_302_Norm_test_statistic(self):
        """Test the normal distribution statistic value is set"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertTrue(NormTest(st.norm.rvs(*parms, size=100), alpha=alpha, display=False).statistic,
                        "FAIL: Normal test statistic not set")

    def test_303_Norm_test_W_value(self):
        """Test the normal distribution W value is set"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertTrue(NormTest(st.norm.rvs(*parms, size=100), alpha=alpha, display=False).w_value,
                        "FAIL: Normal test W value not set")

    def test_304_Norm_test_multi_pass(self):
        """Test if multiple vectors are from the normal distribution"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), st.norm.rvs(4, 0.75, size=75), st.norm.rvs(1, 1, size=50)]
        self.assertGreater(NormTest(*groups, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Normal test Type I error")

    def test_305_Norm_test_multi_fail(self):
        """Test if multiple vectors are from the normal distribution, with one failing"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), st.weibull_min.rvs(1.7, size=75), st.norm.rvs(1, 1, size=50)]
        self.assertLess(NormTest(*groups, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Normal test Type II error")


if __name__ == '__main__':
    unittest.main()
