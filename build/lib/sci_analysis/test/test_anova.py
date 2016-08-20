import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import Anova, MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    def test_550_ANOVA_matched(self):
        """Test the ANOVA class on matched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertGreater(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: ANOVA Type I error")

    def test_551_ANOVA_matched_statistic(self):
        """Test the ANOVA class on matched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertAlmostEqual(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).statistic,
                               0.1076,
                               delta=0.0001,
                               msg="FAIL: ANOVA statistic")

    def test_552_ANOVA_matched_f_value(self):
        """Test the ANOVA class on matched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertAlmostEqual(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).f_value,
                               0.1076,
                               delta=0.0001,
                               msg="FAIL: ANOVA f value")

    def test_553_ANOVA_unmatched(self):
        """Test the ANOVA class on unmatched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        y_parms = [6, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertLess(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: ANOVA Type II error")

    def test_554_ANOVA_matched_just_above_min_size(self):
        """Test the ANOVA class on matched data just above min size"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=3)
        y_input_array = st.norm.rvs(*x_parms, size=3)
        z_input_array = st.norm.rvs(*x_parms, size=3)
        alpha = 0.05
        self.assertTrue(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                        "FAIL: ANOVA just above minimum size")

    def test_555_ANOVA_matched_just_at_size(self):
        """Test the ANOVA class on matched data at min size"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=2)
        y_input_array = st.norm.rvs(*x_parms, size=2)
        z_input_array = st.norm.rvs(*x_parms, size=2)
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: Anova(x_input_array, y_input_array, z_input_array,
                                                          alpha=alpha,
                                                          display=False).p_value)

    def test_556_ANOVA_matched_single_empty_vector(self):
        """Test the ANOVA class on matched data with a single empty vector"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = ["one", "two", "three", "four", "five"]
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertGreater(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: Test should pass with single empty vector")

    def test_557_ANOVA_matched_all_empty_vectors(self):
        """Test the ANOVA class on matched data with all vectors empty"""
        np.random.seed(987654321)
        x_input_array = [float("nan"), float("nan"), float("nan"), "four", float("nan")]
        y_input_array = ["one", "two", "three", "four", "five"]
        alpha = 0.05
        self.assertRaises(NoDataError, lambda: Anova(x_input_array,
                                                     y_input_array,
                                                     alpha=alpha,
                                                     display=False).p_value)


if __name__ == '__main__':
    unittest.main()
