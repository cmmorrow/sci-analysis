import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import Anova


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
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array = st.norm.rvs(*y_parms, size=1000)
        z_input_array = st.norm.rvs(*x_parms, size=1000)
        alpha = 0.05
        self.assertLess(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: ANOVA Type II error")


if __name__ == '__main__':
    unittest.main()
