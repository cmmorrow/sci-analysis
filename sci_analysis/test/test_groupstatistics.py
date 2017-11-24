import unittest
import numpy as np
import pandas as pd
import scipy.stats as st

from ..analysis import GroupStatistics, GroupStatisticsStacked
from ..analysis.exc import MinimumSizeError, NoDataError
from ..data import Vector


class TestGroupStatistics(unittest.TestCase):
    def test_0001_group_statistics_no_name(self):
        """Test the Group Statistic class with generated group names"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       1             
45             2.3678        3.5551       -4.8034        2.2217        11.4199      2             
18             8.0944        1.1855        6.0553        7.9712        10.5272      3             """
        res = GroupStatistics(x_input_array, y_input_array, z_input_array, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)

    def test_0002_group_statistics_group_names(self):
        """Test the Group Statistic class with group names specified in a list"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        names = ("one", "two", "three")
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       one           
18             8.0944        1.1855        6.0553        7.9712        10.5272      three         
45             2.3678        3.5551       -4.8034        2.2217        11.4199      two           """
        res = GroupStatistics(x_input_array, y_input_array, z_input_array, groups=names, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)

    def test_0003_group_statistics_dict(self):
        """Test the Group Statistic class with data passed as a dict"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertTrue(GroupStatistics(data, display=False),
                        "FAIL: Could not display group statistics with passed dict")

    def test_0004_group_statistics_dict_just_above_min_size(self):
        """Test the Group Statistic class with data passed as a dict just above min size"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=2)
        y_input_array = st.norm.rvs(2, 3, size=2)
        z_input_array = st.norm.rvs(8, 1, size=2)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertTrue(GroupStatistics(data, display=False),
                        "FAIL: Just above min size")

    def test_0005_group_statistics_dict_at_min_size(self):
        """Test the Group Statistic class with data passed as a dict at min size"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=1)
        y_input_array = st.norm.rvs(2, 3, size=1)
        z_input_array = st.norm.rvs(8, 1, size=1)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        self.assertRaises(MinimumSizeError, lambda: GroupStatistics(data, display=False))

    def test_0006_group_statistics_dict_single_empty_vector(self):
        """Test the Group Statistic class with data passed as a dict and a single missing vector"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=10)
        y_input_array = ["this", "is", "a", "string"]
        z_input_array = st.norm.rvs(8, 1, size=10)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
10             2.3511        1.3732        0.6591        2.3882        4.2466       one           
10             7.9466        1.0927        6.3630        7.9607        9.7260       three         """
        res = GroupStatistics(data, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)

    def test_0007_group_statistics_dict_empty(self):
        """Test the Group Statistic class with data passed as empty"""
        np.random.seed(987654321)
        x_input_array = ["this", "is", "a", "string"]
        y_input_array = [float("nan"), float("nan"), "three", float("nan")]
        data = {"one": x_input_array, "two": y_input_array}
        self.assertRaises(NoDataError, lambda: GroupStatistics(data, display=False))

    def test_0008_group_statistics_dict_empty_zero_length(self):
        """Test the Group Statistic class with data passed as zero length vectors"""
        np.random.seed(987654321)
        x_input_array = np.array([])
        y_input_array = []
        data = {"one": x_input_array, "two": y_input_array}
        self.assertRaises(NoDataError, lambda: GroupStatistics(data, display=False))

    def test_0009_group_statistics_stacked(self):
        """Test the Stacked Group Statistic class"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        vals = np.append(x_input_array, np.append(y_input_array, z_input_array)).tolist()
        grps = ['x'] * 100 + ['y'] * 45 + ['z'] * 18
        ref = pd.DataFrame({'values': vals, 'groups': grps})
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       x             
45             2.3678        3.5551       -4.8034        2.2217        11.4199      y             
18             8.0944        1.1855        6.0553        7.9712        10.5272      z             """
        res = GroupStatisticsStacked(ref['values'], ref['groups'], display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)

    def test_0010_group_statistics_stacked_no_data(self):
        """Test the event when all passed data is NA"""
        input_array = [np.nan, np.nan, np.nan, np.nan, np.nan]
        grps = ['a', 'b', 'a', 'b', 'a']
        ref = pd.DataFrame({'values': input_array, 'groups': grps})
        self.assertRaises(NoDataError, lambda: GroupStatisticsStacked(ref['values'], ref['groups'], display=False))

    def test_0011_group_statistics_stacked_scalar(self):
        """Test the event a scalar is passed and a minimum size error is raised"""
        input_array = 1
        grps = 'a'
        self.assertRaises(MinimumSizeError, lambda: GroupStatisticsStacked(input_array, grps, display=False))

    def test_0012_group_statistics_stacked_missing_group(self):
        """Test the event when a group is all NA"""
        input_array = [1.0, np.nan, 0.95, np.nan, 1.05]
        grps = ['a', 'b', 'a', 'b', 'a']
        ref = pd.DataFrame({'values': input_array, 'groups': grps})
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
3              1.0000        0.0500        0.9500        1.0000        1.0500       a             """
        res = GroupStatisticsStacked(ref['values'], ref['groups'], display=True)
        self.assertTrue(res)
        self.assertEqual(str(res), output)

    def test_0013_group_statistics_stacked_vector(self):
        """Test the Stacked Group Statistic class with a Vector input object."""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        vals = np.append(x_input_array, np.append(y_input_array, z_input_array)).tolist()
        grps = ['x'] * 100 + ['y'] * 45 + ['z'] * 18
        ref = pd.DataFrame({'values': vals, 'groups': grps})
        exp = Vector(ref['values'], groups=ref['groups'])
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       x             
45             2.3678        3.5551       -4.8034        2.2217        11.4199      y             
18             8.0944        1.1855        6.0553        7.9712        10.5272      z             """
        res = GroupStatisticsStacked(exp, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)

    def test_0014_group_statistics_stacked_missing_groups(self):
        """Test the case where the groups argument is not provided."""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        vals = np.append(x_input_array, np.append(y_input_array, z_input_array)).tolist()
        self.assertRaises(AttributeError, lambda: GroupStatisticsStacked(vals))


if __name__ == '__main__':
    unittest.main()
