import unittest
import scipy.stats as st
import numpy as np

from ..analysis import KSTest
from ..analysis.exc import MinimumSizeError, NoDataError


class MyTestCase(unittest.TestCase):
    def test_250_Kolmogorov_Smirnov_normal_test(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        test = KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False)
        output = """

Kolmogorov-Smirnov Test
-----------------------

alpha   =  0.0500
D value =  0.0584
p value =  0.8853

H0: Data is matched to the norm distribution
"""
        self.assertGreater(test.p_value, alpha)
        self.assertEqual(str(test), output)

    def test_251_Kolmogorov_Smirnov_normal_test_distribution_type(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertEqual(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).distribution, distro)

    def test_252_Kolmogorov_Smirnov_normal_test_statistic(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertTrue(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).statistic)

    def test_253_Kolmogorov_Smirnov_normal_test_D_value(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertTrue(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).d_value)

    def test_254_Kolmogorov_Smirnov_alpha_test_parms_missing(self):
        """Test the KSTest to make sure an exception is raised if parms are missing"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'alpha'  # not to be confused with the sensitivity alpha
        self.assertRaises(TypeError, lambda: KSTest(st.alpha.rvs(size=100), distro, alpha=alpha, display=False))

    def test_255_Kolmogorov_Smirnov_alpha_test(self):
        """Test the alpha distribution detection"""
        np.random.seed(987654321)
        parms = [3.5]
        alpha = 0.05
        distro = 'alpha'
        self.assertGreater(KSTest(st.alpha.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha)

    def test_256_Kolmogorov_Smirnov_beta_test(self):
        """Test the beta distribution detection"""
        np.random.seed(987654321)
        parms = [2.3, 0.6]
        alpha = 0.05
        distro = 'beta'
        self.assertGreater(KSTest(st.beta.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha)

    def test_257_Kolmogorov_Smirnov_cauchy_test(self):
        """Test the cauchy distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'cauchy'
        self.assertGreater(KSTest(st.cauchy.rvs(size=100), distro,
                                  alpha=alpha, display=False).p_value, alpha)

    def test_258_Kolmogorov_Smirnov_chi2_large_test(self):
        """Test the chi squared distribution detection with sufficiently large dof"""
        np.random.seed(987654321)
        parms = [50]
        alpha = 0.05
        distro = 'chi2'
        self.assertGreater(KSTest(st.chi2.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha)

    def test_259_Kolmogorov_Smirnov_chi2_small_test(self):
        """Test the chi squared distribution detection with small dof"""
        np.random.seed(987654321)
        parms = [5]
        alpha = 0.05
        distro = 'chi2'
        self.assertGreater(KSTest(st.chi2.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha)

    def test_260_Kolmogorov_Smirnov_weibull_min_test(self):
        """Test the weibull min distribution detection"""
        np.random.seed(987654321)
        parms = [1.7]
        alpha = 0.05
        distro = 'weibull_min'
        self.assertGreater(KSTest(st.weibull_min.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha)

    def test_261_Kolmogorov_Smirnov_weibull_max_test(self):
        """Test the weibull min distribution detection"""
        np.random.seed(987654321)
        parms = [2.8]
        alpha = 0.05
        distro = 'weibull_max'
        self.assertGreater(KSTest(st.weibull_max.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha)

    def test_262_Kolmogorov_Smirnov_normal_test_at_min_size(self):
        """Test the normal distribution detection at the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertRaises(MinimumSizeError, lambda: KSTest(st.norm.rvs(size=2),
                                                           distro,
                                                           alpha=alpha,
                                                           display=False).p_value)

    def test_263_Kolmogorov_Smirnov_normal_test_just_above_min_size(self):
        """Test the normal distribution detection just above the minimum size"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertTrue(KSTest(st.norm.rvs(size=3), distro, alpha=alpha, display=False).p_value)

    def test_264_Kolmogorov_Smirnov_normal_test_empty_vector(self):
        """Test the normal distribution detection with an empty vector"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertRaises(NoDataError, lambda: KSTest(["one", "two", "three", "four"],
                                                      distro,
                                                      alpha=alpha,
                                                      display=False).p_value)


if __name__ == '__main__':
    unittest.main()
