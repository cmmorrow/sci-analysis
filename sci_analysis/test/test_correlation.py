import unittest
import numpy as np
import scipy.stats as st

from ..analysis import Correlation
from ..analysis.exc import MinimumSizeError, NoDataError
from ..data import UnequalVectorLengthError, Vector


class MyTestCase(unittest.TestCase):
    def test_Correlation_corr_pearson(self):
        """Test the Correlation class for correlated normally distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in x_input_array])
        alpha = 0.05
        output = """

Pearson Correlation Coefficient
-------------------------------

alpha   =  0.0500
r value =  0.8904
p value =  0.0000

HA: There is a significant relationship between predictor and response
"""
        exp = Correlation(x_input_array, y_input_array, alpha=alpha, display=False)
        self.assertLess(exp.p_value, alpha, "FAIL: Correlation pearson Type II error")
        self.assertEqual(exp.test_type, 'pearson')
        self.assertAlmostEqual(exp.r_value, 0.8904, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, 0.8904, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_Correlation_no_corr_pearson(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.norm.rvs(size=100)
        alpha = 0.05
        output = """

Pearson Correlation Coefficient
-------------------------------

alpha   =  0.0500
r value = -0.0055
p value =  0.9567

H0: There is no significant relationship between predictor and response
"""
        exp = Correlation(x_input_array, y_input_array, alpha=alpha, display=False)
        self.assertGreater(exp.p_value, alpha, "FAIL: Correlation pearson Type I error")
        self.assertEqual(exp.test_type, 'pearson')
        self.assertAlmostEqual(exp.r_value, -0.0055, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, -0.0055, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.9567, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_Correlation_corr_spearman(self):
        """Test the Correlation class for correlated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.weibull_min.rvs(1.7, size=100))
        y_input_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in x_input_array])
        alpha = 0.05
        output = """

Spearman Correlation Coefficient
--------------------------------

alpha   =  0.0500
r value =  0.7271
p value =  0.0000

HA: There is a significant relationship between predictor and response
"""
        exp = Correlation(x_input_array, y_input_array, alpha=alpha, display=False)
        self.assertLess(exp.p_value, alpha, "FAIL: Correlation spearman Type II error")
        self.assertEqual(exp.test_type, 'spearman')
        self.assertAlmostEqual(exp.r_value, 0.7271, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, 0.7271, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_Correlation_no_corr_spearman(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        output = """

Spearman Correlation Coefficient
--------------------------------

alpha   =  0.0500
r value = -0.0528
p value =  0.6021

H0: There is no significant relationship between predictor and response
"""
        exp = Correlation(x_input_array, y_input_array, alpha=alpha, display=False)
        self.assertGreater(exp.p_value, alpha, "FAIL: Correlation spearman Type I error")
        self.assertEqual(exp.test_type, 'spearman')
        self.assertAlmostEqual(exp.r_value, -0.0528, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.6021, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, -0.0528, delta=0.0001)
        self.assertTrue(np.array_equal(x_input_array, exp.xdata))
        self.assertTrue(np.array_equal(x_input_array, exp.predictor))
        self.assertTrue(np.array_equal(y_input_array, exp.ydata))
        self.assertTrue(np.array_equal(y_input_array, exp.response))
        self.assertEqual(str(exp), output)

    def test_Correlation_no_corr_pearson_just_above_min_size(self):
        """Test the Correlation class for uncorrelated normally distributed data just above the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertTrue(Correlation(st.norm.rvs(size=4),
                                    st.norm.rvs(size=4),
                                    alpha=alpha,
                                    display=False).p_value,
                        "FAIL: Correlation pearson just above minimum size")

    def test_Correlation_no_corr_pearson_at_min_size(self):
        """Test the Correlation class for uncorrelated normally distributed data at the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: Correlation(st.norm.rvs(size=3),
                                                                st.norm.rvs(size=3),
                                                                alpha=alpha,
                                                                display=False).p_value)

    def test_Correlation_no_corr_pearson_unequal_vectors(self):
        """Test the Correlation class for uncorrelated normally distributed data with unequal vectors"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=87)
        y_input_array = st.norm.rvs(size=100)
        self.assertRaises(UnequalVectorLengthError, lambda: Correlation(x_input_array, y_input_array,
                                                                        alpha=alpha,
                                                                        display=False).p_value)

    def test_Correlation_no_corr_pearson_empty_vector(self):
        """Test the Correlation class for uncorrelated normally distributed data with an empty vector"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: Correlation(["one", "two", "three", "four", "five"],
                                                           st.norm.rvs(size=5),
                                                           alpha=alpha,
                                                           display=False).p_value)

    def test_Correlation_vector(self):
        """Test the Correlation class with an input Vector"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in x_input_array])
        alpha = 0.05
        output = """

Pearson Correlation Coefficient
-------------------------------

alpha   =  0.0500
r value =  0.8904
p value =  0.0000

HA: There is a significant relationship between predictor and response
"""
        exp = Correlation(Vector(x_input_array, other=y_input_array), alpha=alpha, display=False)
        self.assertLess(exp.p_value, alpha, "FAIL: Correlation pearson Type II error")
        self.assertEqual(exp.test_type, 'pearson')
        self.assertAlmostEqual(exp.r_value, 0.8904, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, 0.8904, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_Correlation_vector_alpha(self):
        """Test the Correlation class with an input Vector and different alpha"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in x_input_array])
        alpha = 0.01
        output = """

Pearson Correlation Coefficient
-------------------------------

alpha   =  0.0100
r value =  0.8904
p value =  0.0000

HA: There is a significant relationship between predictor and response
"""
        exp = Correlation(Vector(x_input_array, other=y_input_array), alpha=alpha, display=False)
        self.assertLess(exp.p_value, alpha, "FAIL: Correlation pearson Type II error")
        self.assertEqual(exp.test_type, 'pearson')
        self.assertAlmostEqual(exp.r_value, 0.8904, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0, delta=0.0001)
        self.assertAlmostEqual(exp.statistic, 0.8904, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_Correlation_missing_ydata(self):
        """Test the case where no ydata is given."""
        np.random.seed(987654321)
        x_input_array = range(1, 101)
        self.assertRaises(AttributeError, lambda: Correlation(x_input_array))


if __name__ == '__main__':
    unittest.main()
