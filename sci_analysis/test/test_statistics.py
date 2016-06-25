import unittest
import numpy as np
import scipy.stats as st

from ..analysis.analysis import VectorStatistics


class MyTestCase(unittest.TestCase):
    def test_1000_Vector_stats_count(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertEqual(VectorStatistics(input_array, sample=True, display=False).count, 100, "FAIL: Stat count")

    def test_1001_Vector_stats_mean(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).mean,
                               4.0145,
                               delta=0.0001,
                               msg="FAIL: Stat mean")

    def test_1002_Vector_stats_std_dev_sample(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).std_dev,
                               1.8622,
                               delta=0.0001,
                               msg="FAIL: Stat std dev")

    def test_1003_Vector_stats_std_dev_population(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=False, display=False).std_dev,
                               1.8529,
                               delta=0.0001,
                               msg="FAIL: Stat std dev")

    def test_1004_Vector_stats_std_error_sample(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).std_err,
                               0.1862,
                               delta=0.0001,
                               msg="FAIL: Stat std error")

    def test_1004_Vector_stats_std_error_population(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=False, display=False).std_err,
                               0.1853,
                               delta=0.0001,
                               msg="FAIL: Stat std error")

    def test_1005_Vector_stats_skewness(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).skewness,
                               -0.0256,
                               delta=0.0001,
                               msg="FAIL: Stat skewness")

    def test_1006_Vector_stats_kurtosis(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).kurtosis,
                               -0.4830,
                               delta=0.0001,
                               msg="FAIL: Stat kurtosis")

    def test_1007_Vector_stats_maximum(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).maximum,
                               7.9315,
                               delta=0.0001,
                               msg="FAIL: Stat maximum")

    def test_1008_Vector_stats_q3(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).q3,
                               5.0664,
                               delta=0.0001,
                               msg="FAIL: Stat q3")

    def test_1009_Vector_stats_median(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).median,
                               4.1331,
                               delta=0.0001,
                               msg="FAIL: Stat median")

    def test_1010_Vector_stats_q1(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).q1,
                               2.6576,
                               delta=0.0001,
                               msg="FAIL: Stat q1")

    def test_1011_Vector_stats_minimum(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).minimum,
                               -0.3256,
                               delta=0.0001,
                               msg="FAIL: Stat minimum")

    def test_1012_Vector_stats_range(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).range,
                               8.2571,
                               delta=0.0001,
                               msg="FAIL: Stat range")

    def test_1013_Vector_stats_iqr(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).iqr,
                               2.4088,
                               delta=0.0001,
                               msg="FAIL: Stat iqr")

    def test_1014_Vector_stats_name(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertEqual(VectorStatistics(input_array, sample=True, display=False).name,
                         "Statistics",
                         "FAIL: Stat name")

    def test_1015_Vector_stats_min_size(self):
        """Test the vector statistics class"""
        input_array = np.array([14])
        self.assertFalse(VectorStatistics(input_array, sample=True, display=False).data, "FAIL: Stats not None")

    def test_1016_Vector_stats_empty_array(self):
        """Test the vector statistics class"""
        self.assertFalse(VectorStatistics(np.array([]), sample=True, display=False).data,
                         "FAIL: Stats not None")


if __name__ == '__main__':
    unittest.main()
