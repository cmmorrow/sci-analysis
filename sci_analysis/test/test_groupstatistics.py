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

Overall Statistics
------------------

Number of Groups =  3
Total            =  163
Grand Mean       =  4.1568
Pooled Std Dev   =  2.0798
Grand Median     =  2.2217


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
        self.assertEqual(res.total, 163)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 2.0798, 4)
        self.assertAlmostEqual(res.pooled_std, 2.0798, 4)
        self.assertAlmostEqual(res.gmean, 4.1568, 4)
        self.assertAlmostEqual(res.grand_mean, 4.1568, 4)
        self.assertAlmostEqual(res.gmedian, 2.2217, 4)
        self.assertAlmostEqual(res.grand_median, 2.2217, 4)

    def test_0002_group_statistics_group_names(self):
        """Test the Group Statistic class with group names specified in a list"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        names = ("one", "two", "three")
        output = """

Overall Statistics
------------------

Number of Groups =  3
Total            =  163
Grand Mean       =  4.1568
Pooled Std Dev   =  2.0798
Grand Median     =  2.2217


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
        output = """

Overall Statistics
------------------

Number of Groups =  3
Total            =  163
Grand Mean       =  4.1568
Pooled Std Dev   =  2.0798
Grand Median     =  2.2217


Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       one           
18             8.0944        1.1855        6.0553        7.9712        10.5272      three         
45             2.3678        3.5551       -4.8034        2.2217        11.4199      two           """
        res = GroupStatistics(data, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 163)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 2.0798, 4)
        self.assertAlmostEqual(res.pooled_std, 2.0798, 4)
        self.assertAlmostEqual(res.gmean, 4.1568, 4)
        self.assertAlmostEqual(res.grand_mean, 4.1568, 4)
        self.assertAlmostEqual(res.gmedian, 2.2217, 4)
        self.assertAlmostEqual(res.grand_median, 2.2217, 4)

    def test_0004_group_statistics_dict_just_above_min_size(self):
        """Test the Group Statistic class with data passed as a dict just above min size"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=2)
        y_input_array = st.norm.rvs(2, 3, size=2)
        z_input_array = st.norm.rvs(8, 1, size=2)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        output = """

Overall Statistics
------------------

Number of Groups =  3
Total            =  6
Grand Mean       =  4.4847
Pooled Std Dev   =  4.0150
Grand Median     =  3.1189


Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
2              2.8003        2.0453        1.3541        2.8003        4.2466       one           
2              7.5349        0.7523        7.0029        7.5349        8.0668       three         
2              3.1189        6.6038       -1.5507        3.1189        7.7885       two           """
        res = GroupStatistics(data, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 6)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 4.0150, 4)
        self.assertAlmostEqual(res.pooled_std, 4.0150, 4)
        self.assertAlmostEqual(res.gmean, 4.4847, 4)
        self.assertAlmostEqual(res.grand_mean, 4.4847, 4)
        self.assertAlmostEqual(res.gmedian, 3.1189, 4)
        self.assertAlmostEqual(res.grand_median, 3.1189, 4)

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

Overall Statistics
------------------

Number of Groups =  2
Total            =  20
Grand Mean       =  5.1489
Pooled Std Dev   =  1.2409
Grand Median     =  5.1744


Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
10             2.3511        1.3732        0.6591        2.3882        4.2466       one           
10             7.9466        1.0927        6.3630        7.9607        9.7260       three         """
        res = GroupStatistics(data, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 20)
        self.assertEqual(res.k, 2)
        self.assertAlmostEqual(res.pooled, 1.2409, 4)
        self.assertAlmostEqual(res.pooled_std, 1.2409, 4)
        self.assertAlmostEqual(res.gmean, 5.1489, 4)
        self.assertAlmostEqual(res.grand_mean, 5.1489, 4)
        self.assertAlmostEqual(res.gmedian, 5.1744, 4)
        self.assertAlmostEqual(res.grand_median, 5.1744, 4)

    def test_0007_group_statistics_single_group(self):
        """Test the Group Statistic class with a single group"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=10)
        output = """

Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
10             2.3511        1.3732        0.6591        2.3882        4.2466       1             """
        res = GroupStatistics(x_input_array, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 10)
        self.assertEqual(res.k, 1)
        self.assertIsNone(res.pooled)
        self.assertIsNone(res.pooled_std)
        self.assertIsNone(res.gmean)
        self.assertIsNone(res.grand_mean)
        self.assertIsNone(res.gmedian)
        self.assertIsNone(res.grand_median)

    def test_0008_group_statistics_dict_empty(self):
        """Test the Group Statistic class with data passed as empty"""
        np.random.seed(987654321)
        x_input_array = ["this", "is", "a", "string"]
        y_input_array = [float("nan"), float("nan"), "three", float("nan")]
        data = {"one": x_input_array, "two": y_input_array}
        self.assertRaises(NoDataError, lambda: GroupStatistics(data, display=False))

    def test_0009_group_statistics_dict_empty_zero_length(self):
        """Test the Group Statistic class with data passed as zero length vectors"""
        np.random.seed(987654321)
        x_input_array = np.array([])
        y_input_array = []
        data = {"one": x_input_array, "two": y_input_array}
        self.assertRaises(NoDataError, lambda: GroupStatistics(data, display=False))

    def test_0010_group_statistics_stacked(self):
        """Test the Stacked Group Statistic class"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        vals = np.append(x_input_array, np.append(y_input_array, z_input_array)).tolist()
        grps = ['x'] * 100 + ['y'] * 45 + ['z'] * 18
        ref = pd.DataFrame({'values': vals, 'groups': grps})
        output = """

Overall Statistics
------------------

Number of Groups =  3
Total            =  163
Grand Mean       =  4.1568
Pooled Std Dev   =  2.0798
Grand Median     =  2.2217


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
        self.assertEqual(res.total, 163)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 2.0798, 4)
        self.assertAlmostEqual(res.pooled_std, 2.0798, 4)
        self.assertAlmostEqual(res.gmean, 4.1568, 4)
        self.assertAlmostEqual(res.grand_mean, 4.1568, 4)
        self.assertAlmostEqual(res.gmedian, 2.2217, 4)
        self.assertAlmostEqual(res.grand_median, 2.2217, 4)

    def test_0011_group_statistics_stacked_no_data(self):
        """Test the event when all passed data is NA"""
        input_array = [np.nan, np.nan, np.nan, np.nan, np.nan]
        grps = ['a', 'b', 'a', 'b', 'a']
        ref = pd.DataFrame({'values': input_array, 'groups': grps})
        self.assertRaises(NoDataError, lambda: GroupStatisticsStacked(ref['values'], ref['groups'], display=False))

    def test_0012_group_statistics_stacked_scalar(self):
        """Test the event a scalar is passed and a minimum size error is raised"""
        input_array = 1
        grps = 'a'
        self.assertRaises(MinimumSizeError, lambda: GroupStatisticsStacked(input_array, grps, display=False))

    def test_0013_group_statistics_stacked_missing_group(self):
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
        res = GroupStatisticsStacked(ref['values'], ref['groups'], display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 3)
        self.assertEqual(res.k, 1)
        self.assertIsNone(res.pooled)
        self.assertIsNone(res.pooled_std)
        self.assertIsNone(res.gmean)
        self.assertIsNone(res.grand_mean)
        self.assertIsNone(res.gmedian)
        self.assertIsNone(res.grand_median)

    def test_0014_group_statistics_stacked_vector(self):
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

Overall Statistics
------------------

Number of Groups =  3
Total            =  163
Grand Mean       =  4.1568
Pooled Std Dev   =  2.0798
Grand Median     =  2.2217


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
        self.assertEqual(res.total, 163)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 2.0798, 4)
        self.assertAlmostEqual(res.pooled_std, 2.0798, 4)
        self.assertAlmostEqual(res.gmean, 4.1568, 4)
        self.assertAlmostEqual(res.grand_mean, 4.1568, 4)
        self.assertAlmostEqual(res.gmedian, 2.2217, 4)
        self.assertAlmostEqual(res.grand_median, 2.2217, 4)

    def test_0015_group_statistics_stacked_missing_groups(self):
        """Test the case where the groups argument is not provided."""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        vals = np.append(x_input_array, np.append(y_input_array, z_input_array)).tolist()
        self.assertRaises(AttributeError, lambda: GroupStatisticsStacked(vals))

    def test_0016_group_statistics_above_min_size(self):
        """Test the Stacked Group Statistic class"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=2)
        vals = np.append(x_input_array, np.append(y_input_array, z_input_array)).tolist()
        grps = ['x'] * 100 + ['y'] * 45 + ['z'] * 2
        ref = pd.DataFrame({'values': vals, 'groups': grps})
        output = """

Overall Statistics
------------------

Number of Groups =  3
Total            =  147
Grand Mean       =  4.8060
Pooled Std Dev   =  2.1549
Grand Median     =  2.2217


Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       x             
45             2.3678        3.5551       -4.8034        2.2217        11.4199      y             
2              10.0420       0.6862        9.5568        10.0420       10.5272      z             """
        res = GroupStatisticsStacked(ref['values'], ref['groups'], display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 147)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 2.1549, 4)
        self.assertAlmostEqual(res.pooled_std, 2.1549, 4)
        self.assertAlmostEqual(res.gmean, 4.8060, 4)
        self.assertAlmostEqual(res.grand_mean, 4.8060, 4)
        self.assertAlmostEqual(res.gmedian, 2.2217, 4)
        self.assertAlmostEqual(res.grand_median, 2.2217, 4)

    def test_0017_group_statistics_dict_groups_is_none(self):
        """Test the Group Statistic class with data passed as a dict and None passed to groups"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(2, 1, size=100)
        y_input_array = st.norm.rvs(2, 3, size=45)
        z_input_array = st.norm.rvs(8, 1, size=18)
        data = {"one": x_input_array, "two": y_input_array, "three": z_input_array}
        output = """

Overall Statistics
------------------

Number of Groups =  3
Total            =  163
Grand Mean       =  4.1568
Pooled Std Dev   =  2.0798
Grand Median     =  2.2217


Group Statistics
----------------

n             Mean          Std Dev       Min           Median        Max           Group         
--------------------------------------------------------------------------------------------------
100            2.0083        1.0641       -0.4718        2.0761        4.2466       one           
18             8.0944        1.1855        6.0553        7.9712        10.5272      three         
45             2.3678        3.5551       -4.8034        2.2217        11.4199      two           """
        res = GroupStatistics(data, groups=None, display=False)
        self.assertTrue(res)
        self.assertEqual(str(res), output)
        self.assertEqual(res.total, 163)
        self.assertEqual(res.k, 3)
        self.assertAlmostEqual(res.pooled, 2.0798, 4)
        self.assertAlmostEqual(res.pooled_std, 2.0798, 4)
        self.assertAlmostEqual(res.gmean, 4.1568, 4)
        self.assertAlmostEqual(res.grand_mean, 4.1568, 4)
        self.assertAlmostEqual(res.gmedian, 2.2217, 4)
        self.assertAlmostEqual(res.grand_median, 2.2217, 4)


if __name__ == '__main__':
    unittest.main()
