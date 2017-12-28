import unittest
import numpy as np
import scipy.stats as st

from ..analysis import NormTest
from ..analysis.exc import MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    def test_300_Norm_test_single(self):
        """Test the normal distribution check"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        x_input = st.norm.rvs(*parms, size=100)
        other = """

Shapiro-Wilk test for normality
-------------------------------

alpha   =  0.0500
W value =  0.9880
p value =  0.5050

H0: Data is normally distributed
"""
        self.assertGreater(NormTest(x_input, display=False, alpha=alpha).p_value, alpha,
                           "FAIL: Normal test Type I error")
        self.assertEqual(str(NormTest(x_input, display=False, alpha=alpha)), other)

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
        self.assertGreater(NormTest(*groups, alpha=alpha, display=True).p_value, alpha,
                           "FAIL: Normal test Type I error")

    def test_305_Norm_test_multi_fail(self):
        """Test if multiple vectors are from the normal distribution, with one failing"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), st.weibull_min.rvs(1.7, size=75), st.norm.rvs(1, 1, size=50)]
        self.assertLess(NormTest(*groups, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Normal test Type II error")

    def test_306_Norm_test_single_just_above_min_size(self):
        """Test the normal distribution at just above the minimum size"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertGreater(NormTest(st.norm.rvs(*parms, size=3), display=False, alpha=alpha).p_value, alpha,
                           "FAIL: Normal test just above the minimum size")

    def test_307_Norm_test_single_at_min_size(self):
        """Test the normal distribution at the minimum size"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: NormTest(st.norm.rvs(*parms, size=2),
                                                             display=False,
                                                             alpha=alpha).p_value)

    def test_308_Norm_test_multi_at_min_size(self):
        """Test if multiple vectors are from the normal distribution at the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=2), st.norm.rvs(4, 0.75, size=10), st.norm.rvs(1, 1, size=50)]
        self.assertRaises(MinimumSizeError, lambda: NormTest(*groups, alpha=alpha, display=False).p_value)

    def test_309_Norm_test_multi_with_single_missing_vector(self):
        """Test if multiple vectors are from the normal distribution with single vector missing"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), ["one", "two", "three", "four"], st.norm.rvs(1, 1, size=50)]
        self.assertTrue(NormTest(*groups, alpha=alpha, display=False).p_value,
                        "FAIL: Normal test with single missing vector")

    def test_310_Norm_test_single_empty(self):
        """Test with empty vector"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: NormTest(["one", "two", "three", "four"],
                                                        alpha=alpha,
                                                        display=False).p_value)

    def test_311_Norm_test_multi_all_empty_vectors(self):
        """Test if multiple vectors are from the normal distribution with all missing vectors"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [[float("nan"), float("nan"), "three", float("nan")], ["one", "two", "three", "four"]]
        self.assertRaises(NoDataError, lambda: NormTest(*groups, alpha=alpha, display=False).p_value)

    def test_312_Norm_test_multi_with_single_scalar(self):
        """Test if multiple vectors are from the normal distribution with single scalar"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), "string", st.norm.rvs(1, 1, size=50)]
        self.assertTrue(NormTest(*groups, alpha=alpha, display=False).p_value,
                        "FAIL: Normal test with single scalar should pass")

    def test_313_Norm_test_multi_with_all_scalar(self):
        """Test if multiple vectors are from the normal distribution with all scalar"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = ["this", "is", "a", "string"]
        self.assertRaises(NoDataError, lambda: NormTest(*groups, alpha=alpha, display=False))


if __name__ == '__main__':
    unittest.main()
