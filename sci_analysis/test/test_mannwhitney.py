import unittest
import numpy as np
import scipy.stats as st

from analysis.analysis import MannWhitney, MinimumSizeError, NoDataError


class TestMannWhitney(unittest.TestCase):
    def test_MannWhitney_matched(self):
        """Test the MannWhitney U test with two matched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertGreater(MannWhitney(st.weibull_min.rvs(*x_parms, size=100),
                                       st.weibull_min.rvs(*y_parms, size=100),
                                       alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: MannWhitney Type I error")

    def test_MannWhitney_unmatched(self):
        """Test the MannWhitney U test with two unmatched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [8.2]
        alpha = 0.05
        self.assertLess(MannWhitney(st.weibull_min.rvs(*x_parms, size=100),
                                    st.weibull_min.rvs(*y_parms, size=100),
                                    alpha=alpha, display=False).p_value,
                        alpha,
                        "FAIL: ManWhitney Type II error")

    def test_MannWhitney_statistic(self):
        """Test the MannWhitney U test statistic"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertAlmostEqual(MannWhitney(st.weibull_min.rvs(*x_parms, size=100),
                                           st.weibull_min.rvs(*y_parms, size=100),
                                           alpha=alpha, display=False).statistic,
                               4976.0,
                               delta=0.0001,
                               msg="FAIL: MannWhitney statistic incorrect")

    def test_MannWhitney_u_value(self):
        """Test the MannWhitney U test u value"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertAlmostEqual(MannWhitney(st.weibull_min.rvs(*x_parms, size=100),
                                           st.weibull_min.rvs(*y_parms, size=100),
                                           alpha=alpha, display=False).u_value,
                               4976.0,
                               delta=0.0001,
                               msg="FAIL: MannWhitney u value incorrect")

    def test_MannWhitney_matched_just_above_min_size(self):
        """Test the MannWhitney U test with matched samples just above minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertGreater(MannWhitney(st.weibull_min.rvs(*x_parms, size=31),
                                       st.weibull_min.rvs(*y_parms, size=31),
                                       alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: MannWhitney matched just above min size")

    def test_MannWhitney_unmatched_just_above_min_size(self):
        """Test the MannWhitney U test with two unmatched samples just above minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [8.2]
        alpha = 0.1
        self.assertLess(MannWhitney(st.weibull_min.rvs(*x_parms, size=50),
                                    st.weibull_min.rvs(*y_parms, size=31),
                                    alpha=alpha, display=True).p_value,
                        alpha,
                        "FAIL: ManWhitney unmatched just above min size")

    def test_MannWhitney_matched_at_min_size(self):
        """Test the MannWhitney U test with matched samples at minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: MannWhitney(st.weibull_min.rvs(*x_parms, size=45),
                                                                st.weibull_min.rvs(*y_parms, size=30),
                                                                alpha=alpha, display=False).p_value)

    def test_MannWhitney_one_missing_array(self):
        """Test the MannWhitney U test with one missing array"""
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: MannWhitney([1.2, 0.9, 1.4, 1.0], ["one", "two", "three", "four"],
                                                                alpha=alpha,
                                                                display=False))

    def test_MannWhitney_two_missing_arrays(self):
        """Test the MannWhitney U test with two missing arrays"""
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: MannWhitney(["five", "six", "seven", "eight"],
                                                           ["one", "two", "three", "four"],
                                                           alpha=alpha,
                                                           display=False))


if __name__ == '__main__':
    unittest.main()
