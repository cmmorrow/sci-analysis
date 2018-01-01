import unittest
import scipy.stats as st
import numpy as np

from ..data import Vector
from ..analysis import TTest
from ..analysis.exc import MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    # Test TTest

    def test_200_TTest_single_matched(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        x_input = st.norm.rvs(*x_parms, size=100)
        y_val = 4.0
        alpha = 0.05
        exp = TTest(x_input, y_val, display=False)
        output = """

1 Sample T Test
---------------

alpha   =  0.0500
t value =  0.0781
p value =  0.9379

H0: Means are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: TTest single type I error")
        self.assertEqual(exp.test_type, '1_sample')
        self.assertEqual(exp.mu, 4.0)
        self.assertAlmostEqual(exp.statistic, 0.0781, delta=0.0001)
        self.assertAlmostEqual(exp.t_value, 0.0781, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.9379, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_205_TTest_single_unmatched(self):
        """Test the TTest against a given unmatched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 5.0
        alpha = 0.05
        x_input = st.norm.rvs(*x_parms, size=100)
        exp = TTest(x_input, y_val, display=False)
        output = """

1 Sample T Test
---------------

alpha   =  0.0500
t value = -12.4518
p value =  0.0000

HA: Means are significantly different
"""
        self.assertFalse(exp.p_value > alpha, "FAIL: TTest single type II error")
        self.assertEqual(exp.mu, 5.0)
        self.assertEqual(exp.test_type, '1_sample')
        self.assertAlmostEqual(exp.statistic, -12.4518, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, -12.4518, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_206_TTest_equal_variance_matched(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        x_input = st.norm.rvs(*x_parms, size=100)
        y_input = st.norm.rvs(*y_parms, size=100)
        alpha = 0.05
        exp = TTest(x_input, y_input, display=False)
        output = """

T Test
------

alpha   =  0.0500
t value = -0.2592
p value =  0.7957

H0: Means are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: TTest equal variance matched Type I error")
        self.assertIsNone(exp.mu)
        self.assertEqual(exp.test_type, 't_test')
        self.assertAlmostEqual(exp.statistic, -0.2592, delta=0.0001)
        self.assertAlmostEqual(exp.t_value, -0.2592, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.7957, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_209_TTest_equal_variance_unmatched(self):
        """Test the TTest with two samples with equal variance and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 0.75]
        x_input = st.norm.rvs(*x_parms, size=100)
        y_input = st.norm.rvs(*y_parms, size=100)
        alpha = 0.05
        exp = TTest(x_input, y_input, display=False)
        output = """

T Test
------

alpha   =  0.0500
t value = -4.6458
p value =  0.0000

HA: Means are significantly different
"""
        self.assertLess(exp.p_value, alpha, "FAIL: TTest equal variance unmatched Type II error")
        self.assertEqual(exp.test_type, 't_test')
        self.assertEqual(str(exp), output)

    def test_210_TTest_unequal_variance_matched(self):
        """Test the TTest with two samples with different variances and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 1.35]
        x_input = st.norm.rvs(*x_parms, size=100)
        y_input = st.norm.rvs(*y_parms, size=100)
        alpha = 0.05
        exp = TTest(x_input, y_input, display=False)
        output = """

Welch's T Test
--------------

alpha   =  0.0500
t value = -0.3487
p value =  0.7278

H0: Means are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: TTest different variance matched Type I error")
        self.assertEqual(exp.test_type, 'welch_t')
        self.assertEqual(str(exp), output)

    def test_211_TTest_unequal_variance_unmatched(self):
        """Test the TTest with two samples with different variances and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 1.12]
        x_input = st.norm.rvs(*x_parms, size=100)
        y_input = st.norm.rvs(*y_parms, size=100)
        alpha = 0.05
        exp = TTest(x_input, y_input, display=True)
        output = """

Welch's T Test
--------------

alpha   =  0.0500
t value = -3.7636
p value =  0.0002

HA: Means are significantly different
"""
        self.assertLess(exp.p_value, alpha, "FAIL: TTest different variance unmatched Type II error")
        self.assertEqual(exp.test_type, 'welch_t')
        self.assertEqual(str(exp), output)

    def test_214_TTest_equal_variance_matched_min_size_above(self):
        """Test the TTest at the minimum size threshold"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        x_input = st.norm.rvs(*x_parms, size=4)
        y_input = st.norm.rvs(*y_parms, size=4)
        alpha = 0.05
        exp = TTest(x_input, y_input, display=False)
        output = """

T Test
------

alpha   =  0.0500
t value =  0.9450
p value =  0.3811

H0: Means are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: TTest minimum size fail")
        self.assertEqual(str(exp), output)

    def test_215_TTest_equal_variance_matched_min_size_below(self):
        """Test the TTest just above the minimum size threshold"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: TTest(st.norm.rvs(*x_parms, size=3),
                                                          st.norm.rvs(*y_parms, size=3),
                                                          alpha=alpha,
                                                          display=False).p_value)

    def test_216_TTest_equal_variance_matched_one_missing_array(self):
        """Test the TTest test with one missing array"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: TTest([1.1, 1.0, 0.9, 0.8],
                                                     ["one", "two", "three", "four"],
                                                     alpha=alpha,
                                                     display=False).p_value)

    def test_217_TTest_with_vector_input(self):
        """Test the TTest test with a vector object."""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        x_input = st.norm.rvs(*x_parms, size=100)
        y_input = st.norm.rvs(*y_parms, size=100)
        vector = Vector(x_input).append(Vector(y_input))
        alpha = 0.05
        exp = TTest(vector, display=False)
        output = """

T Test
------

alpha   =  0.0500
t value = -0.2592
p value =  0.7957

H0: Means are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: TTest equal variance matched Type I error")
        self.assertIsNone(exp.mu)
        self.assertEqual(exp.test_type, 't_test')
        self.assertAlmostEqual(exp.statistic, -0.2592, delta=0.0001)
        self.assertAlmostEqual(exp.t_value, -0.2592, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.7957, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_217_TTest_with_missing_second_arg(self):
        """Test the case where the second argument is None."""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        x_input = st.norm.rvs(*x_parms, size=100)
        self.assertRaises(AttributeError, lambda: TTest(x_input))


if __name__ == '__main__':
    unittest.main()
