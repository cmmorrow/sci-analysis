import unittest
import numpy as np
import scipy.stats as st

from ..data import Vector
from ..analysis import MannWhitney
from ..analysis.exc import MinimumSizeError, NoDataError


class TestMannWhitney(unittest.TestCase):
    def test_MannWhitney_matched(self):
        """Test the MannWhitney U test with two matched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=100)
        y_input = st.weibull_min.rvs(*y_parms, size=100)
        alpha = 0.05
        exp = MannWhitney(x_input, y_input, alpha=alpha, display=True)
        output = """

Mann Whitney U Test
-------------------

alpha   =  0.0500
u value =  5024.0000
p value =  1.0477

H0: Locations are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: MannWhitney Type I error")
        self.assertAlmostEqual(exp.statistic, 5024.0, delta=0.0001, msg="FAIL: MannWhitney statistic incorrect")
        self.assertAlmostEqual(exp.u_value, 5024.0, delta=0.0001, msg="FAIL: MannWhitney u_value incorrect")
        self.assertAlmostEqual(exp.p_value, 1.0477, delta=0.0001, msg="FAIL: MannWhitney p_value incorrect")
        self.assertEqual(str(exp), output)

    def test_MannWhitney_unmatched(self):
        """Test the MannWhitney U test with two unmatched samples"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [8.2]
        x_input = st.weibull_min.rvs(*x_parms, size=100)
        y_input = st.weibull_min.rvs(*y_parms, size=100)
        alpha = 0.05
        exp = MannWhitney(x_input, y_input, alpha=alpha, display=False)
        output = """

Mann Whitney U Test
-------------------

alpha   =  0.0500
u value =  4068.0000
p value =  0.0228

HA: Locations are not matched
"""
        self.assertLess(exp.p_value, alpha, msg="FAIL: ManWhitney Type II error")
        self.assertAlmostEqual(exp.statistic, 4068.0, delta=0.0001)
        self.assertAlmostEqual(exp.u_value, 4068.0, delta=0.0001)
        self.assertAlmostEqual(exp.p_value, 0.0228, delta=0.0001)
        self.assertEqual(str(exp), output)

    def test_MannWhitney_matched_just_above_min_size(self):
        """Test the MannWhitney U test with matched samples just above minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=21)
        y_input = st.weibull_min.rvs(*y_parms, size=21)
        alpha = 0.05
        exp = MannWhitney(x_input, y_input, alpha=alpha, display=False)
        output = """

Mann Whitney U Test
-------------------

alpha   =  0.0500
u value =  222.0000
p value =  1.0401

H0: Locations are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: MannWhitney matched just above min size")
        self.assertEqual(str(exp), output)

    def test_MannWhitney_unmatched_just_above_min_size(self):
        """Test the MannWhitney U test with two unmatched samples just above minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [8.2]
        x_input = st.weibull_min.rvs(*x_parms, size=50)
        y_input = st.weibull_min.rvs(*y_parms, size=21)
        alpha = 0.1
        exp = MannWhitney(x_input, y_input, alpha=alpha, display=False)
        output = """

Mann Whitney U Test
-------------------

alpha   =  0.1000
u value =  440.0000
p value =  0.2871

H0: Locations are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: ManWhitney unmatched just above min size")
        self.assertEqual(str(exp), output)

    def test_MannWhitney_matched_at_min_size(self):
        """Test the MannWhitney U test with matched samples at minimum size"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=45)
        y_input = st.weibull_min.rvs(*y_parms, size=20)
        alpha = 0.05
        self.assertRaises(MinimumSizeError, lambda: MannWhitney(x_input, y_input, alpha=alpha, display=False))

    def test_MannWhitney_one_missing_array(self):
        """Test the MannWhitney U test with one missing array"""
        x_input = [np.random.randint(1, 50) for _ in range(50)]
        y_input = ['abcdefghijklmnop'[:np.random.randint(1, 17)] for _ in range(50)]
        self.assertRaises(NoDataError, lambda: MannWhitney(x_input, y_input, display=False))

    def test_MannWhitney_two_missing_arrays(self):
        """Test the MannWhitney U test with two missing arrays"""
        x_input = ['abcdefghijklmnop'[:np.random.randint(1, 17)] for _ in range(50)]
        y_input = ['abcdefghijklmnop'[:np.random.randint(1, 17)] for _ in range(50)]
        self.assertRaises(NoDataError, lambda: MannWhitney(x_input, y_input, display=False))

    def test_MannWhitney_vector_input(self):
        """Test the case where the input argument is a Vector object."""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=100)
        y_input = st.weibull_min.rvs(*y_parms, size=100)
        vector = Vector(x_input).append(Vector(y_input))
        alpha = 0.05
        exp = MannWhitney(vector, alpha=alpha, display=True)
        output = """

Mann Whitney U Test
-------------------

alpha   =  0.0500
u value =  5024.0000
p value =  1.0477

H0: Locations are matched
"""
        self.assertGreater(exp.p_value, alpha, "FAIL: MannWhitney Type I error")
        self.assertAlmostEqual(exp.statistic, 5024.0, delta=0.0001, msg="FAIL: MannWhitney statistic incorrect")
        self.assertAlmostEqual(exp.u_value, 5024.0, delta=0.0001, msg="FAIL: MannWhitney u_value incorrect")
        self.assertAlmostEqual(exp.p_value, 1.0477, delta=0.0001, msg="FAIL: MannWhitney p_value incorrect")
        self.assertEqual(str(exp), output)

    def test_MannWhitney_missing_second_arg(self):
        """Test the case where the second argument is missing."""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input = st.weibull_min.rvs(*x_parms, size=100)
        self.assertRaises(AttributeError, lambda: MannWhitney(x_input))


if __name__ == '__main__':
    unittest.main()
