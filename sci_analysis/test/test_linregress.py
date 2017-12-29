import unittest
import numpy as np
import scipy.stats as st

from ..analysis import LinearRegression
from ..analysis.exc import MinimumSizeError, NoDataError
from ..data import UnequalVectorLengthError, Vector


class MyTestCase(unittest.TestCase):
    def test_350_LinRegress_corr(self):
        """Test the Linear Regression class for correlation"""
        np.random.seed(987654321)
        x_input_array = range(1, 101)
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        output = """

Linear Regression
-----------------

n         =  100
Slope     =  3.0000
Intercept =  0.0000
r         =  1.0000
r^2       =  1.0000
Std Err   =  0.0000
p value   =  0.0000
"""
        self.assertLess(LinearRegression(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Linear Regression Type II error")
        self.assertEqual(str(LinearRegression(x_input_array, y_input_array, alpha=alpha, display=False)), output)

    def test_351_LinRegress_no_corr(self):
        """Test the Linear Regression class for uncorrelated data"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertGreater(LinearRegression(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Linear Regression Type I error")

    def test_352_LinRegress_no_corr_slope(self):
        """Test the Linear Regression slope"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).slope, -0.0969, delta=0.0001,
                               msg="FAIL: Linear Regression slope")

    def test_353_LinRegress_no_corr_intercept(self):
        """Test the Linear Regression intercept"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).intercept, -0.0397, delta=0.0001,
                               msg="FAIL: Linear Regression intercept")

    def test_354_LinRegress_no_corr_r(self):
        """Test the Linear Regression r"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).r_value, -0.1029, delta=0.0001,
                               msg="FAIL: Linear Regression r")

    def test_355_LinRegress_no_corr_r2(self):
        """Test the Linear Regression r^2"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).r_squared, 0.0105, delta=0.0001,
                               msg="FAIL: Linear Regression r^2")

    def test_356_LinRegress_no_corr_std_err(self):
        """Test the Linear Regression std err"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).std_err, 0.0666, delta=0.0001,
                               msg="FAIL: Linear Regression std err")

    def test_357_LinRegress_no_corr_just_above_min_size(self):
        """Test the Linear Regression class for uncorrelated data just above minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=4)
        y_input_array = st.norm.rvs(size=4)
        self.assertTrue(LinearRegression(x_input_array, y_input_array, alpha=alpha, display=False).p_value,
                        "FAIL: Linear Regression just above minimum size")

    def test_358_LinRegress_no_corr_at_min_size(self):
        """Test the Linear Regression class for uncorrelated data at minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=3)
        y_input_array = st.norm.rvs(size=3)
        self.assertRaises(MinimumSizeError, lambda: LinearRegression(x_input_array, y_input_array,
                                                                     alpha=alpha,
                                                                     display=False).p_value)

    def test_359_LinRegress_no_corr_unequal_vectors(self):
        """Test the Linear Regression class for uncorrelated data with unequal vectors"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=184)
        y_input_array = st.norm.rvs(size=200)
        self.assertRaises(UnequalVectorLengthError, lambda: LinearRegression(x_input_array, y_input_array,
                                                                             alpha=alpha,
                                                                             display=False).p_value)

    def test_360_LinRegress_no_corr_empty_vector(self):
        """Test the Linear Regression class for uncorrelated data with an empty vector"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = [float("nan"), "two", "three", "four", float("nan")]
        y_input_array = st.norm.rvs(size=5)
        self.assertRaises(NoDataError, lambda: LinearRegression(x_input_array, y_input_array,
                                                                alpha=alpha,
                                                                display=False).p_value)

    def test_361_LinRegress_no_corr_two_empty_vectors(self):
        """Test the Linear Regression class for uncorrelated data with two empty vectors"""
        alpha = 0.05
        x_input_array = [float("nan"), "two", "three", "four", float("nan")]
        y_input_array = ["one", "two", float("nan"), "four", float("nan")]
        self.assertRaises(NoDataError, lambda: LinearRegression(x_input_array, y_input_array,
                                                                alpha=alpha,
                                                                display=False).p_value)

    def test_362_LinRegress_no_corr_statistic(self):
        """Test the Linear Regression R^2"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).statistic, 0.0105, delta=0.0001,
                               msg="FAIL: Linear Regression statistic")

    def test_363_LinRegress_vector(self):
        """Test the Linear Regression class with an input Vector."""
        np.random.seed(987654321)
        x_input_array = range(1, 101)
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        output = """

Linear Regression
-----------------

n         =  100
Slope     =  3.0000
Intercept =  0.0000
r         =  1.0000
r^2       =  1.0000
Std Err   =  0.0000
p value   =  0.0000
"""
        exp = LinearRegression(Vector(x_input_array, other=y_input_array), alpha=alpha, display=False)
        self.assertLess(exp.p_value, alpha, "FAIL: Linear Regression Type II error")
        self.assertEqual(str(exp), output)

    def test_364_LinRegress_missing_ydata(self):
        """Test the case where no ydata is given."""
        np.random.seed(987654321)
        x_input_array = range(1, 101)
        self.assertRaises(AttributeError, lambda: LinearRegression(x_input_array))


if __name__ == '__main__':
    unittest.main()
