import unittest
import numpy as np
import scipy.stats as st

from analysis.analysis import TwoSampleKSTest, MinimumSizeError, NoDataError


class TestTwoSampleKS(unittest.TestCase):
    def test_two_sample_KS_matched(self):
        """Test the Two Sample KS Test with matched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertGreater(TwoSampleKSTest(st.weibull_min.rvs(*x_parms, size=20),
                                           st.weibull_min.rvs(*y_parms, size=20),
                                           alpha=alpha,
                                           display=False).p_value,
                           alpha,
                           "FAIL: Two Sample KS Test Type I error")

    def test_two_sample_KS_unmatched(self):
        """Test the Two Sample KS Test with unmatched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [8.2]
        alpha = 0.06
        self.assertLess(TwoSampleKSTest(st.weibull_min.rvs(*x_parms, size=20),
                                        st.weibull_min.rvs(*y_parms, size=20),
                                        alpha=alpha,
                                        display=False).p_value,
                        alpha,
                        "FAIL: Two Sample KS Test Type II error")

    def test_two_sample_KS_statistic(self):
        """Test the Two Sample KS Test test statistic"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertAlmostEqual(TwoSampleKSTest(st.weibull_min.rvs(*x_parms, size=20),
                                               st.weibull_min.rvs(*y_parms, size=20),
                                               alpha=alpha,
                                               display=False).statistic,
                               0.2,
                               delta=0.1,
                               msg="FAIL: Two Sample KS Test statistic")

    def test_two_sample_KS_d_value(self):
        """Test the Two Sample KS Test test d value"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertAlmostEqual(TwoSampleKSTest(st.weibull_min.rvs(*x_parms, size=20),
                                               st.weibull_min.rvs(*y_parms, size=20),
                                               alpha=alpha,
                                               display=False).statistic,
                               0.2,
                               delta=0.1,
                               msg="FAIL: Two Sample KS Test d value")

    def test_two_sample_KS_matched_at_min_size(self):
        """Test the Two Sample KS Test with matched samples at the minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: TwoSampleKSTest(st.weibull_min.rvs(*x_parms, size=2),
                                                                    st.weibull_min.rvs(*y_parms, size=2),
                                                                    alpha=alpha,
                                                                    display=False).p_value)

    def test_two_sample_KS_matched_just_above_min_size(self):
        """Test the Two Sample KS Test with matched samples just above the minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertTrue(TwoSampleKSTest(st.weibull_min.rvs(*x_parms, size=3),
                                        st.weibull_min.rvs(*y_parms, size=3),
                                        alpha=alpha,
                                        display=False).p_value,
                        "FAIL: Just above the min value")

    def test_two_sample_KS_matched_empty(self):
        """Test the Two Sample KS Test with empty vectors"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: TwoSampleKSTest([float("nan"), float("nan"), "one", float("nan")],
                                                               ["one", "two", "three", "four"],
                                                               alpha=alpha,
                                                               display=False).p_value)


if __name__ == '__main__':
    unittest.main()
