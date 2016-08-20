import unittest
import numpy as np
import scipy.stats as st

from analysis.analysis import GroupStatistics, MinimumSizeError, NoDataError


class TestGroupStatistics(unittest.TestCase):
    def test_0001_group_statistics_no_name(self):
        """Test the Group Statistic class with generated group names"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        self.assertTrue(GroupStatistics(x_input_array, y_input_array, z_input_array, display=True),
                        "FAIL: Could not display group statistics with generated group names")

    def test_0002_group_statistics_group_names(self):
        """Test the Group Statistic class with group names specified in a list"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        names = ("one", "two", "three")
        self.assertTrue(GroupStatistics(x_input_array, y_input_array, z_input_array, groups=names, display=True),
                        "FAIL: Could not display group statistics with passed group names")

    def test_0003_group_statistics_dict(self):
        """Test the Group Statistic class with data passed as a dict"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertTrue(GroupStatistics(data, display=True),
                        "FAIL: Could not display group statistics with passed dict")

    def test_0004_group_statistics_dict_just_above_min_size(self):
        """Test the Group Statistic class with data passed as a dict just above min size"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=2)
        y_input_array = st.norm.rvs(2, 3, size=2)
        z_input_array = st.norm.rvs(8, 1, size=2)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertTrue(GroupStatistics(data, display=True),
                        "FAIL: Just above min size")

    def test_0005_group_statistics_dict_at_min_size(self):
        """Test the Group Statistic class with data passed as a dict at min size"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=1)
        y_input_array = st.norm.rvs(2, 3, size=1)
        z_input_array = st.norm.rvs(8, 1, size=1)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertRaises(MinimumSizeError, lambda: GroupStatistics(data, display=True))

    def test_0006_group_statistics_dict_single_empty_vector(self):
        """Test the Group Statistic class with data passed as a dict and a single missing vector"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=10)
        y_input_array = ["this", "is", "a", "string"]
        z_input_array = st.norm.rvs(8, 1, size=10)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertTrue(GroupStatistics(data, display=True), "FAIL: Should pass with a single missing vector")

    def test_0007_group_statistics_dict_empty(self):
        """Test the Group Statistic class with data passed as empty"""
        np.random.seed(987654321)
        x_input_array = ["this", "is", "a", "string"]
        y_input_array = [float("nan"), float("nan"), "three", float("nan")]
        data = {"one": x_input_array, "two": y_input_array}
        self.assertRaises(NoDataError, lambda: GroupStatistics(data, display=True))

    def test_0008_group_statistics_dict_empty_zero_length(self):
        """Test the Group Statistic class with data passed as zero length vectors"""
        np.random.seed(987654321)
        x_input_array = np.array([])
        y_input_array = []
        data = {"one": x_input_array, "two": y_input_array}
        self.assertRaises(NoDataError, lambda: GroupStatistics(data, display=True))


if __name__ == '__main__':
    unittest.main()
