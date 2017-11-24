import unittest
import numpy as np
import scipy.stats as st

from ..analysis import Kruskal
from ..analysis.exc import MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    def test_500_Kruskal_matched(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        alpha = 0.05
        exp = Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False)
        output = """

Kruskal-Wallis
--------------

alpha   =  0.0500
h value =  0.4042
p value =  0.8170

H0: Group means are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: Kruskal Type I error")
        self.assertAlmostEqual(exp.statistic, 0.4042, delta=0.0001)
        self.assertAlmostEqual(exp.h_value, 0.4042, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.817, delta=0.001)
        self.assertEqual(str(exp), output)

    def test_503_Kruskal_matched_single_argument(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(NoDataError, lambda: Kruskal(x_input_array, alpha=a, display=False).p_value)

    def test_504_Kruskal_unmatched(self):
        """Test the Kruskal Wallis class on unmatched data"""
        np.random.seed(987654321)
        x_parms = [1.7, 1]
        z_parms = [0.8, 1]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        alpha = 0.05
        exp = Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False)
        output = """

Kruskal-Wallis
--------------

alpha   =  0.0500
h value =  37.4069
p value =  0.0000

HA: Group means are not matched
"""
        self.assertLess(exp.p_value, alpha, "FAIL: Kruskal Type II error")
        self.assertAlmostEqual(exp.statistic, 37.4069, delta=0.0001)
        self.assertAlmostEqual(exp.h_value, 37.4069, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_505_Kruskal_matched_just_above_min_size(self):
        """Test the Kruskal Wallis class on matched data just above min size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=3)
        y_input_array = st.weibull_min.rvs(*x_parms, size=3)
        z_input_array = st.weibull_min.rvs(*x_parms, size=3)
        alpha = 0.05
        exp = Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False)
        output = """

Kruskal-Wallis
--------------

alpha   =  0.0500
h value =  3.4667
p value =  0.1767

H0: Group means are matched
"""
        self.assertGreater(exp.p_value, alpha)
        self.assertEqual(str(exp), output)

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
        exp = Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False)
        output = """

Kruskal-Wallis
--------------

alpha   =  0.0500
h value =  0.0034
p value =  0.9532

H0: Group means are matched
"""
        self.assertGreater(exp.p_value, alpha)
        self.assertEqual(str(exp), output)

    def test_508_Kruskal_matched_all_empty(self):
        """Test the Kruskal Wallis class on matched data all empty"""
        np.random.seed(987654321)
        x_input_array = [np.nan, np.nan, np.nan, "four", np.nan]
        y_input_array = ["one", "two", "three", "four", "five"]
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: Kruskal(x_input_array, y_input_array,
                                                       alpha=alpha,
                                                       display=False).p_value)


if __name__ == '__main__':
    unittest.main()
