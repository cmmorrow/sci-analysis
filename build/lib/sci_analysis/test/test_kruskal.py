import unittest
import numpy as np
import scipy.stats as st

from analysis.analysis import Kruskal, MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    def test_500_Kruskal_matched(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertGreater(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: Kruskal Type I error")

    def test_501_Kruskal_matched_statistic(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(Kruskal(x_input_array, y_input_array, z_input_array, alpha=a, display=False).statistic,
                               0.4042,
                               delta=0.0001,
                               msg="FAIL: Kruskal statistic")

    def test_502_Kruskal_matched_h_value(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(Kruskal(x_input_array, y_input_array, z_input_array, alpha=a, display=False).h_value,
                               0.4042,
                               delta=0.0001,
                               msg="FAIL: Kruskal h value")

    def test_503_Kruskal_matched_single_argument(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(TypeError, lambda: Kruskal(x_input_array, alpha=a, display=False).p_value)

    def test_504_Kruskal_unmatched(self):
        """Test the Kruskal Wallis class on unmatched data"""
        np.random.seed(987654321)
        x_parms = [1.7, 1]
        z_parms = [0.8, 1]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        alpha = 0.05
        self.assertLess(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                        alpha,
                        "FAIL: Kruskal Type II error")

    def test_505_Kruskal_matched_just_above_min_size(self):
        """Test the Kruskal Wallis class on matched data just above min size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=3)
        y_input_array = st.weibull_min.rvs(*x_parms, size=3)
        z_input_array = st.weibull_min.rvs(*x_parms, size=3)
        alpha = 0.05
        self.assertTrue(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                        "FAIL: Kruskal just above min size")

    def test_506_Kruskal_matched_at_min_size(self):
        """Test the Kruskal Wallis class on matched data at min size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=2)
        y_input_array = st.weibull_min.rvs(*x_parms, size=2)
        z_input_array = st.weibull_min.rvs(*x_parms, size=2)
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: Kruskal(x_input_array, y_input_array, z_input_array,
                                                            alpha=alpha,
                                                            display=False).p_value)

    def test_507_Kruskal_matched_single_empty_vector(self):
        """Test the Kruskal Wallis class on matched data with single missing vector"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = ["one", "two", "three", "four", "five"]
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertGreater(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: Kruskal test should pass with single empty vector")

    def test_508_Kruskal_matched_all_empty(self):
        """Test the Kruskal Wallis class on matched data all empty"""
        np.random.seed(987654321)
        x_input_array = [float("nan"), float("nan"), float("nan"), "four", float("nan")]
        y_input_array = ["one", "two", "three", "four", "five"]
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: Kruskal(x_input_array, y_input_array,
                                                       alpha=alpha,
                                                       display=False).p_value)


if __name__ == '__main__':
    unittest.main()
