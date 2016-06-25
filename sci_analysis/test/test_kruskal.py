import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import Kruskal


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


if __name__ == '__main__':
    unittest.main()
