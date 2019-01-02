import unittest
import numpy as np
import scipy.stats as st
from os import path, getcwd
import pandas as pd
from ..graphs import GraphScatter
from ..data import Vector
from ..analysis.exc import NoDataError
from ..data import UnequalVectorLengthError


class MyTestCase(unittest.TestCase):

    @property
    def save_path(self):
        if getcwd().split('/')[-1] == 'test':
            return './images/'
        elif getcwd().split('/')[-1] == 'sci_analysis':
            if path.exists('./setup.py'):
                return './sci_analysis/test/images/'
            else:
                return './test/images/'
        else:
            './'

    def test_100_default(self):
        """Generate a scatter plot with default settings"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     save_to='{}test_scatter_100'.format(self.save_path)))

    def test_101_no_points(self):
        """Generate a scatter plot with no points"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     points=False,
                                     save_to='{}test_scatter_101'.format(self.save_path)))

    def test_102_no_points_contours(self):
        """Generate a scatter plot with no points and contours"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     points=False,
                                     contours=True,
                                     save_to='{}test_scatter_102'.format(self.save_path)))

    def test_103_no_points_contours_boxplots(self):
        """Generate a scatter plot with no points, contours and boxplots"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     points=False,
                                     contours=True,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_103'.format(self.save_path)))

    def test_104_no_fit(self):
        """Generate a scatter plot with no fit"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     fit=False,
                                     save_to='{}test_scatter_104'.format(self.save_path)))

    def test_105_no_fit_no_points(self):
        """Generate a scatter plot with no fit or points"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     fit=False,
                                     points=False,
                                     save_to='{}test_scatter_105'.format(self.save_path)))

    def test_106_no_fit_no_points_contours(self):
        """Generate a scatter plot with no fit or points and contours"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     fit=False,
                                     points=False,
                                     contours=True,
                                     save_to='{}test_scatter_106'.format(self.save_path)))

    def test_107_no_fit_no_points_contours_boxplots(self):
        """Generate a scatter plot with no fit or points, contours and boxplots"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     fit=False,
                                     points=False,
                                     contours=True,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_107'.format(self.save_path)))

    def test_108_contours(self):
        """Generate a scatter plot with contours"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     contours=True,
                                     save_to='{}test_scatter_108'.format(self.save_path)))

    def test_109_contours_boxplots(self):
        """Generate a scatter plot with contours and boxplots"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     contours=True,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_109'.format(self.save_path)))

    def test_110_boxplots(self):
        """Generate a scatter plot with boxplots"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_110'.format(self.save_path)))

    def test_111_no_points_boxplots(self):
        """Generate a scatter plot with no points and boxplots"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     points=False,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_111'.format(self.save_path)))

    def test_112_no_points_no_fit_boxplots(self):
        """Generate a scatter plot with no points or no fit and boxplots"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     fit=False,
                                     points=False,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_112'.format(self.save_path)))

    def test_113_one_empty_list(self):
        """Catch the case where the input is an empty list"""
        np.random.seed(987654321)
        input_x_array = []
        input_y_array = st.norm.rvs(size=2000)
        self.assertRaises(NoDataError, lambda: GraphScatter(input_x_array, input_y_array))

    def test_114_other_empty_list(self):
        """Catch the case where the input is an empty list"""
        np.random.seed(987654321)
        input_y_array = []
        input_x_array = st.norm.rvs(size=2000)
        self.assertRaises(UnequalVectorLengthError, lambda: GraphScatter(input_x_array, input_y_array))

    def test_115_two_empty_lists(self):
        """Catch the case where both inputs are empty lists"""
        np.random.seed(987654321)
        input_x_array = []
        input_y_array = []
        self.assertRaises(NoDataError, lambda: GraphScatter(input_x_array, input_y_array))

    def test_116_missing_data(self):
        """Catch the case where there is missing data in both arrays"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        indicies_x = list(np.random.randint(0, 1999, 200))
        indicies_y = list(np.random.randint(0, 1999, 200))
        for i in indicies_x:
            input_x_array = np.insert(input_x_array, i, np.nan, axis=0)
        for i in indicies_y:
            input_y_array = np.insert(input_y_array, i, np.nan, axis=0)
        self.assertTrue(GraphScatter(input_x_array, input_y_array, save_to='{}test_scatter_116'.format(self.save_path)))

    def test_117_at_min_size(self):
        """Generate a scatter plot at the min size"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2)
        input_y_array = st.norm.rvs(size=2)
        self.assertTrue(GraphScatter(input_x_array, input_y_array, save_to='{}test_scatter_117'.format(self.save_path)))

    def test_118_single_point(self):
        """Generate a scatter plot with a single point"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=1)
        input_y_array = st.norm.rvs(size=1)
        self.assertTrue(GraphScatter(input_x_array, input_y_array, save_to='{}test_scatter_118'.format(self.save_path)))

    def test_119_default_corr(self):
        """Generate a scatter plot with correlating data"""
        np.random.seed(987654321)
        input_x_array = st.weibull_min.rvs(2, size=2000)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertTrue(GraphScatter(input_x_array, input_y_array, save_to='{}test_scatter_119'.format(self.save_path)))

    def test_120_contours_no_fit_corr(self):
        """Generate a scatter plot with contours, no fit and correlating data"""
        np.random.seed(987654321)
        input_x_array = st.weibull_min.rvs(2, size=2000)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     fit=False,
                                     contours=True,
                                     save_to='{}test_scatter_120'.format(self.save_path)))

    def test_121_boxplots_fit_corr(self):
        """Generate a scatter plot with boxplots, fit and correlating data"""
        np.random.seed(987654321)
        input_x_array = st.weibull_min.rvs(2, size=2000)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_121'.format(self.save_path)))

    def test_122_set_x_and_y_name(self):
        """Generate a scatter plot with set x and y names"""
        np.random.seed(987654321)
        input_x_array = st.weibull_min.rvs(2, size=2000)
        input_y_array = np.array([x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     xname='Test X',
                                     yname='Test Y',
                                     save_to='{}test_scatter_122'.format(self.save_path)))

    def test_123_scatter_string(self):
        """Generate a scatter plot from lists of string values"""
        np.random.seed(987654321)
        input_x_array = ["1.0", "2.4", "three", "4", "5.1", "six", "7.3"]
        input_y_array = ["1.2", "2", "3.0", "4.3", "five", "six", "7.8"]
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     # fit=False,
                                     # contours=True,
                                     boxplot_borders=True,
                                     save_to='{}test_scatter_123'.format(self.save_path)))

    def test_124_scatter_length_4_bug(self):
        """Generate a scatter plot with 4 points to check for the case where the scatter method thinks the color
           tuple is a cmap instead of an RGBA tuple"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=4)
        input_y_array = st.norm.rvs(size=4)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     save_to='{}test_scatter_124'.format(self.save_path)))

    def test_125_scatter_title(self):
        """Generate a scatter plot with a specified title"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     title='Test Title',
                                     save_to='{}test_scatter_125'.format(self.save_path)))

    def test_126_scatter_2dim_arrays(self):
        """Generate a scatter plot a 2dim arrays"""
        np.random.seed(987654321)
        input_x_array = np.array([[1, 2, 3], [4, 5, 6]])
        input_y_array = np.array([[3, 6, 9], [12, 15, 18]])
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     title='2dim Arrays',
                                     save_to='{}test_scatter_126'.format(self.save_path)))

    def test_127_scatter_2dim_lists_with_missing(self):
        """Generate a scatter plot with 2dim arrays with missing data"""
        np.random.seed(987654321)
        input_x_array = [['1', '2', 'three'], ['4.0', 'five', '6']]
        input_y_array = [['3', '6', '9'], ['four', 'five', '18.0']]
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     title='2dim Arrays With Missing',
                                     save_to='{}test_scatter_127'.format(self.save_path)))

    def test_128_scatter_3dim_arrays(self):
        """Generate a scatter plot a 3dim arrays"""
        np.random.seed(987654321)
        input_x_array = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        input_y_array = np.array([[[3, 6, 9], [12, 15, 18]], [[21, 24, 27], [30, 33, 36]]])
        self.assertTrue(GraphScatter(input_x_array, input_y_array,
                                     title='3dim Arrays',
                                     save_to='{}test_scatter_128'.format(self.save_path)))

    def test_129_scatter_missing_ydata(self):
        """Check to make sure an AttributeError is raised is ydata is None."""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = None
        self.assertRaises(AttributeError, lambda: GraphScatter(input_x_array, input_y_array))

    def test_130_scatter_vector(self):
        """Generate a scatter plot with a Vector object."""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        vector = Vector(input_x_array, other=input_y_array)
        self.assertTrue(GraphScatter(vector,
                                     title='Vector Scatter',
                                     save_to='{}test_scatter_130'.format(self.save_path)))

    def test_131_scatter_empty_vector(self):
        """Check to make sure a NoDataError exception is raised if an empty Vector is passed."""
        vector = Vector()
        self.assertRaises(NoDataError, lambda: GraphScatter(vector))

    def test_132_negative_corr(self):
        """Generate a scatter plot with correlating data"""
        np.random.seed(987654321)
        input_x_array = st.weibull_min.rvs(2, size=2000)
        input_y_array = np.array([3 - x + st.norm.rvs(0, 0.5, size=1) for x in input_x_array])
        self.assertTrue(GraphScatter(input_x_array, input_y_array, save_to='{}test_scatter_132'.format(self.save_path)))

    def test_133_missing_data_labels(self):
        """Test labels where there's missing data in both arrays"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        input_labels_array = np.random.choice(list('ABCDE'),  size=2000)
        xmask = np.random.randint(0, 2, size=input_x_array.shape).astype(np.bool)
        ymask = np.random.randint(0, 2, size=input_y_array.shape).astype(np.bool)
        input_x_array[xmask] = np.nan
        input_y_array[ymask] = np.nan
        self.assertTrue(
            GraphScatter(
                input_x_array,
                input_y_array,
                labels=input_labels_array,
                highlight=['A'],
                save_to='{}test_scatter_133'.format(self.save_path)
            )
        )

    def test_134_Graphscatter_labels(self):
        """Generates graphscatter with labels but no highlights."""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        input_labels_array = np.random.choice(list('ABCDE'),  size=2000)
        self.assertTrue(
            GraphScatter(
                input_x_array,
                input_y_array,
                labels=input_labels_array,
                save_to='{}test_scatter_134'.format(self.save_path)
            )
        )

    def test_135_Graphscatter_highlights(self):
        """Generates graphscatter with highlights"""
        np.random.seed(987654321)
        input_x_array = st.norm.rvs(size=2000)
        input_y_array = st.norm.rvs(size=2000)
        input_labels_array = np.random.choice(list('ABCDE'),  size=2000)
        self.assertTrue(
            GraphScatter(
                input_x_array,
                input_y_array,
                labels=input_labels_array,
                highlight=['E'],
                save_to='{}test_scatter_135'.format(self.save_path)
            )
        )

    def test_136_Graphscatter_dataframe(self):
        """tests graphscater with dataframe input"""
        np.random.seed(987654321)
        df = pd.DataFrame(np.random.randn(100, 2), columns=list('xy'))
        df['labels'] = np.random.randint(10000, 50000, size=100)
        self.assertTrue(
            GraphScatter(
                df['x'],
                df['y'],
                labels=df['labels'],
                highlight=pd.Series([39870, 44751, 39037, 39742, 45290]),
                save_to='{}test_scatter_136'.format(self.save_path)
            )
        )


if __name__ == '__main__':
    unittest.main()
