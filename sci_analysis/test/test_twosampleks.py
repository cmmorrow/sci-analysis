import unittest
import numpy as np
import scipy.stats as st

from ..data import Vector
from ..analysis import TwoSampleKSTest
from ..analysis.exc import MinimumSizeError, NoDataError


class TestTwoSampleKS(unittest.TestCase):
    def test_two_sample_KS_matched(self):
        """Test the Two Sample KS Test with matched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=20)
        y_input = st.weibull_min.rvs(*y_parms, size=20)
        alpha = 0.05
        exp = TwoSampleKSTest(x_input, y_input, alpha=alpha, display=False)
        output = """

Two Sample Kolmogorov-Smirnov Test
----------------------------------

alpha   =  0.0500
D value =  0.2000
p value =  0.7710

H0: Both samples come from the same distribution
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: Two Sample KS Test Type I error")
        self.assertEqual(str(exp), output)

    def test_two_sample_KS_unmatched(self):
        """Test the Two Sample KS Test with unmatched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [8.2]
        x_input = st.weibull_min.rvs(*x_parms, size=20)
        y_input = st.weibull_min.rvs(*y_parms, size=20)
        alpha = 0.06
        exp = TwoSampleKSTest(x_input, y_input, alpha=alpha, display=False)
        output = """

Two Sample Kolmogorov-Smirnov Test
----------------------------------

alpha   =  0.0600
D value =  0.4000
p value =  0.0591

HA: Samples do not come from the same distribution
"""
        self.assertLess(exp.p_value, alpha, "FAIL: Two Sample KS Test Type II error")
        self.assertEqual(str(exp), output)

    def test_two_sample_KS_statistic(self):
        """Test the Two Sample KS Test test statistic"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=20)
        y_input = st.weibull_min.rvs(*y_parms, size=20)
        alpha = 0.05
        exp = TwoSampleKSTest(x_input, y_input, alpha=alpha, display=False)
        self.assertAlmostEqual(exp.statistic, 0.2, delta=0.1, msg="FAIL: Two Sample KS Test statistic")
        self.assertAlmostEqual(exp.d_value, 0.2, delta=0.1, msg="FAIL: Two Sample KS Test d_value")
        self.assertAlmostEqual(exp.p_value, 0.771, delta=0.001, msg="FAIL: Two Sample KS Test p_value")

    def test_two_sample_KS_matched_at_min_size(self):
        """Test the Two Sample KS Test with matched samples at the minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=2)
        y_input = st.weibull_min.rvs(*y_parms, size=2)
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: TwoSampleKSTest(x_input, y_input, alpha=alpha, display=False))

    def test_two_sample_KS_matched_just_above_min_size(self):
        """Test the Two Sample KS Test with matched samples just above the minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=3)
        y_input = st.weibull_min.rvs(*y_parms, size=3)
        alpha = 0.05
        exp = TwoSampleKSTest(x_input, y_input, alpha=alpha, display=True)
        output = """

Two Sample Kolmogorov-Smirnov Test
----------------------------------

alpha   =  0.0500
D value =  0.6667
p value =  0.3197

H0: Both samples come from the same distribution
"""
        self.assertAlmostEqual(exp.p_value, 0.3197, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, 0.6667, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_two_sample_KS_matched_empty(self):
        """Test the Two Sample KS Test with empty vectors"""
        np.random.seed(987654321)
        x_input = [np.nan, np.nan, "one", np.nan]
        y_input = ["one", "two", "three", "four"]
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: TwoSampleKSTest(x_input, y_input, alpha=alpha, display=False))

    def test_two_sample_KS_vector_input(self):
        """Test the Two Sample KS Test with a Vector object."""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=20)
        y_input = st.weibull_min.rvs(*y_parms, size=20)
        vector = Vector(x_input).append(Vector(y_input))
        alpha = 0.05
        exp = TwoSampleKSTest(vector, alpha=alpha, display=False)
        output = """

Two Sample Kolmogorov-Smirnov Test
----------------------------------

alpha   =  0.0500
D value =  0.2000
p value =  0.7710

H0: Both samples come from the same distribution
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: Two Sample KS Test Type I error")
        self.assertEqual(str(exp), output)

    def test_two_sample_KS_with_missing_second_arg(self):
        """Test the case where the second argument is None."""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=20)
        self.assertRaises(AttributeError, lambda: TwoSampleKSTest(x_input))


if __name__ == '__main__':
    unittest.main()
