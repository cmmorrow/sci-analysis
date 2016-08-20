import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import Correlation, MinimumSizeError, NoDataError
from ..data.data import UnequalVectorLengthError


class MyTestCase(unittest.TestCase):
    def test_400_Correlation_corr_pearson(self):
        """Test the Correlation class for correlated normally distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertLess(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Correlation pearson Type II error")

    def test_401_Correlation_corr_pearson_test_type(self):
        """Test the Correlation class for correlated normally distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertEqual(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).test_type, 'pearson',
                         "FAIL: Correlation pearson wrong type")

    def test_402_Correlation_no_corr_pearson(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertGreater(Correlation(st.norm.rvs(size=100),
                                       st.norm.rvs(size=100),
                                       alpha=alpha,
                                       display=False).p_value, alpha,
                           "FAIL: Correlation pearson Type I error")

    def test_403_Correlation_no_corr_pearson_test_type(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertEqual(Correlation(st.norm.rvs(size=100),
                                     st.norm.rvs(size=100),
                                     alpha=alpha,
                                     display=False).test_type, 'pearson',
                         "FAIL: Correlation pearson wrong type")

    def test_404_Correlation_no_corr_pearson_r_value(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertAlmostEqual(Correlation(st.norm.rvs(size=100),
                                           st.norm.rvs(size=100),
                                           alpha=alpha,
                                           display=False).r_value, -0.0055,
                               delta=0.0001,
                               msg="FAIL: Correlation pearson r value")

    def test_405_Correlation_corr_spearman(self):
        """Test the Correlation class for correlated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.weibull_min.rvs(1.7, size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertLess(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Correlation spearman Type II error")

    def test_406_Correlation_corr_spearman_test_type(self):
        """Test the Correlation class for correlated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.weibull_min.rvs(1.7, size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertEqual(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).test_type, 'spearman',
                         "FAIL: Correlation spearman wrong type")

    def test_407_Correlation_no_corr_spearman(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertGreater(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Correlation spearman Type I error")

    def test_408_Correlation_no_corr_spearman_test_type(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertEqual(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).test_type, 'spearman',
                         "FAIL: Correlation spearman wrong type")

    def test_409_Correlation_no_corr_spearman_xdata(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).xdata,
                                       x_input_array),
                        "FAIL: Correlation spearman xdata")

    def test_410_Correlation_no_corr_spearman_predictor(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).predictor,
                                       x_input_array),
                        "FAIL: Correlation spearman predictor")

    def test_411_Correlation_no_corr_spearman_ydata(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).ydata,
                                       y_input_array),
                        "FAIL: Correlation spearman ydata")

    def test_412_Correlation_no_corr_spearman_response(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).response,
                                       y_input_array),
                        "FAIL: Correlation spearman response")

    def test_413_Correlation_no_corr_pearson_just_above_min_size(self):
        """Test the Correlation class for uncorrelated normally distributed data just above the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertTrue(Correlation(st.norm.rvs(size=4),
                                    st.norm.rvs(size=4),
                                    alpha=alpha,
                                    display=False).p_value,
                        "FAIL: Correlation pearson just above minimum size")

    def test_414_Correlation_no_corr_pearson_at_min_size(self):
        """Test the Correlation class for uncorrelated normally distributed data at the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: Correlation(st.norm.rvs(size=3),
                                                                st.norm.rvs(size=3),
                                                                alpha=alpha,
                                                                display=False).p_value)

    def test_415_Correlation_no_corr_pearson_unequal_vectors(self):
        """Test the Correlation class for uncorrelated normally distributed data with unequal vectors"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=87)
        y_input_array = st.norm.rvs(size=100)
        self.assertRaises(UnequalVectorLengthError, lambda: Correlation(x_input_array, y_input_array,
                                                                        alpha=alpha,
                                                                        display=False).p_value)

    def test_416_Correlation_no_corr_pearson_empty_vector(self):
        """Test the Correlation class for uncorrelated normally distributed data with an empty vector"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: Correlation(["one", "two", "three", "four", "five"],
                                                           st.norm.rvs(size=5),
                                                           alpha=alpha,
                                                           display=False).p_value)


if __name__ == '__main__':
    unittest.main()
