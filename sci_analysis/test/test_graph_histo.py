import unittest
import numpy as np
import scipy.stats as st
from os import path, getcwd

from ..data import Vector
from ..graphs import GraphHisto
from ..analysis.exc import NoDataError


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

    def test_100_default_graph(self):
        """Generate a histogram graph with default arguments"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   save_to='{}test_histo_100'.format(self.save_path)))

    def test_101_bins(self):
        """Generate a histogram graph with 100 bins"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   bins=100,
                                   save_to='{}test_histo_101'.format(self.save_path)))

    def test_102_bins_no_box_plot(self):
        """Generate a histogram graph without the accompanying boxplot"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   bins=100,
                                   boxplot=False,
                                   save_to='{}test_histo_102'.format(self.save_path)))

    def test_103_bins_no_box_plot_cdf(self):
        """Generate a histogram graph with cdf and no boxplot"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   bins=100,
                                   boxplot=False,
                                   cdf=True,
                                   save_to='{}test_histo_103'.format(self.save_path)))

    def test_104_bins_no_box_plot_cdf_fit(self):
        """Generate a histogram graph with fit, cdf and no boxplot"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   bins=100,
                                   boxplot=False,
                                   cdf=True,
                                   fit=True,
                                   save_to='{}test_histo_104'.format(self.save_path)))

    def test_105_no_box_plot(self):
        """Generate a histogram graph without the accompanying boxplot"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   boxplot=False,
                                   save_to='{}test_histo_105'.format(self.save_path)))

    def test_106_no_box_plot_cdf(self):
        """Generate a histogram graph with cdf and no boxplot"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   boxplot=False,
                                   cdf=True,
                                   save_to='{}test_histo_106'.format(self.save_path)))

    def test_107_no_box_plot_cdf_fit(self):
        """Generate a histogram graph with fit, cdf and no boxplot"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   boxplot=False,
                                   cdf=True,
                                   fit=True,
                                   save_to='{}test_histo_107'.format(self.save_path)))

    def test_108_cdf(self):
        """Generate a histogram graph with cdf"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   cdf=True,
                                   save_to='{}test_histo_108'.format(self.save_path)))

    def test_109_cdf_fit(self):
        """Generate a histogram graph with fit and cdf"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   cdf=True,
                                   fit=True,
                                   save_to='{}test_histo_109'.format(self.save_path)))

    def test_110_fit(self):
        """Generate a histogram graph with fit"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   fit=True,
                                   save_to='{}test_histo_110'.format(self.save_path)))

    def test_111_only_mean(self):
        """Generate a histogram graph with only the mean set"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        mean = np.mean(input_array)
        self.assertTrue(GraphHisto(input_array,
                                   mean=mean,
                                   save_to='{}test_histo_111'.format(self.save_path)))

    def test_112_only_std(self):
        """Generate a histogram graph with only the std dev set"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        std = np.std(input_array)
        self.assertTrue(GraphHisto(input_array,
                                   std_dev=std,
                                   save_to='{}test_histo_112'.format(self.save_path)))

    def test_113_mean_and_std(self):
        """Generate a histogram graph with the mean and std dev set"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        mean = np.mean(input_array)
        std = np.std(input_array)
        self.assertTrue(GraphHisto(input_array,
                                   mean=mean,
                                   std_dev=std,
                                   save_to='{}test_histo_113'.format(self.save_path)))

    def test_114_mean_std_and_sample(self):
        """Generate a histogram graph with the mean and std dev set"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        mean = np.mean(input_array)
        std = np.std(input_array)
        self.assertTrue(GraphHisto(input_array,
                                   mean=mean,
                                   std_dev=std,
                                   sample=True,
                                   save_to='{}test_histo_114'.format(self.save_path)))

    def test_115_distribution(self):
        """Generate a histogram graph with distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   save_to='{}test_histo_115'.format(self.save_path)))

    def test_116_distribution_bins(self):
        """Generate a histogram graph with distribution and bins set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   bins=100,
                                   save_to='{}test_histo_116'.format(self.save_path)))

    def test_117_distribution_bins_no_boxplot(self):
        """Generate a histogram graph with no boxplot, cdf, distribution and bins set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   bins=100,
                                   boxplot=False,
                                   save_to='{}test_histo_117'.format(self.save_path)))

    def test_118_distribution_bins_boxplot_cdf(self):
        """Generate a histogram graph with no boxplot, distribution and bins set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   bins=100,
                                   boxplot=False,
                                   cdf=True,
                                   save_to='{}test_histo_118'.format(self.save_path)))

    def test_119_distribution_bins_boxplot_cdf_fit(self):
        """Generate a histogram graph with no boxplot, fit, cdf, distribution and bins set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   bins=100,
                                   boxplot=False,
                                   cdf=True,
                                   fit=True,
                                   save_to='{}test_histo_119'.format(self.save_path)))

    def test_120_distribution_boxplot(self):
        """Generate a histogram graph with no boxplot and distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   boxplot=False,
                                   save_to='{}test_histo_120'.format(self.save_path)))

    def test_121_distribution_boxplot_cdf(self):
        """Generate a histogram graph with no boxplot, cdf and distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   boxplot=False,
                                   cdf=True,
                                   save_to='{}test_histo_121'.format(self.save_path)))

    def test_122_distribution_boxplot_cdf_fit(self):
        """Generate a histogram graph with no boxplot, fit, cdf and distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   boxplot=False,
                                   cdf=True,
                                   fit=True,
                                   save_to='{}test_histo_122'.format(self.save_path)))

    def test_123_distribution_cdf(self):
        """Generate a histogram graph with cdf and distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   cdf=True,
                                   save_to='{}test_histo_123'.format(self.save_path)))

    def test_124_distribution_cdf_fit(self):
        """Generate a histogram graph with fit, cdf and distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   cdf=True,
                                   fit=True,
                                   save_to='{}test_histo_124'.format(self.save_path)))

    def test_125_distribution_fit(self):
        """Generate a histogram graph with fit and distribution set"""
        np.random.seed(987654321)
        input_array = st.weibull_min.rvs(1.7, size=5000)
        self.assertTrue(GraphHisto(input_array,
                                   distribution='weibull_min',
                                   fit=True,
                                   save_to='{}test_histo_125'.format(self.save_path)))

    def test_126_empty_list(self):
        """Catch the graphing case where the input is an empty list"""
        np.random.seed(987654321)
        input_array = []
        self.assertRaises(NoDataError, lambda: GraphHisto(input_array))

    def test_127_empty_array(self):
        """Catch the graphing case where the input is an empty array"""
        np.random.seed(987654321)
        input_array = np.array([])
        self.assertRaises(NoDataError, lambda: GraphHisto(input_array))

    def test_128_xname(self):
        """Set the xname of a histogram graph"""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=5000))
        self.assertTrue(GraphHisto(input_array,
                                   xname='Test',
                                   save_to='{}test_histo_128'.format(self.save_path)))

    def test_129_name(self):
        """Set the name of a histogram graph"""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=5000))
        self.assertTrue(GraphHisto(input_array,
                                   name='Test',
                                   save_to='{}test_histo_129'.format(self.save_path)))

    def test_130_yname(self):
        """Set the yname of a histogram graph"""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=5000))
        self.assertTrue(GraphHisto(input_array,
                                   yname='Test',
                                   save_to='{}test_histo_130'.format(self.save_path)))

    def test_131_missing_data(self):
        """Generate a histogram graph with 500 random missing values"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=5000)
        indicies = list(np.random.randint(0, 4999, 500))
        for x in indicies:
            input_array = np.insert(input_array, x, np.nan, axis=0)
        self.assertTrue(GraphHisto(Vector(input_array), name='Missing Test',
                                   save_to='{}test_histo_131'.format(self.save_path)))

    # def test_132_at_min_size(self):
    #     """Generate a histogram graph at the minimum size"""
    #     np.random.seed(987654321)
    #     input_array = Vector(st.norm.rvs(size=2))
    #     self.assertTrue(GraphHisto(input_array, name='At Min Size', save_to='{}test_histo_132'.format(self.save_path)))
    #
    # def test_133_min_size(self):
    #     """Generate a histogram graph below the minimum size"""
    #     np.random.seed(987654321)
    #     input_array = Vector(st.norm.rvs(size=1))
    #     self.assertRaises(MinimumSizeError, lambda: GraphHisto(input_array))

    def test_134_graph_string(self):
        """Generate a histogram graph with string data"""
        np.random.seed(987654321)
        input_array = ["1", "2", "this", "is", "a", '3', "string", "4", "5"]
        self.assertTrue(GraphHisto(input_array, name='String Array', save_to='{}test_histo_134'.format(self.save_path)))

    def test_135_graph_2dim_array(self):
        """Generate a histogram graph with a 2dim array"""
        np.random.seed(987654321)
        input_array = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(GraphHisto(input_array, name='2dim Array', save_to='{}test_histo_135'.format(self.save_path)))

    def test_136_graph_3dim_array(self):
        """Generate a histogram graph with a 3dim array"""
        np.random.seed(987654321)
        input_array = np.array([[[1, 2, 3], [4, 5, 6]], [[10, 11, 12], [13, 14, 15]]])
        self.assertTrue(GraphHisto(input_array, name='3dim Array', save_to='{}test_histo_136'.format(self.save_path)))

    def test_137_graph_3dim_missing_data(self):
        """Generate a histogram graph from a 3dim list with missing data"""
        np.random.seed(987654321)
        input_array = [[['1', '2', 'three'], ['4.0', 'five', '6']], [['10', '11', '12.00'], ['t', 'h', '15']]]
        self.assertTrue(GraphHisto(input_array, name='3dim Missing', save_to='{}test_histo_137'.format(self.save_path)))

    def test_138_graph_title(self):
        """Generate a histogram graph with a specified title"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=2000)
        self.assertTrue(GraphHisto(input_array, title='Title Test', save_to='{}test_histo_138'.format(self.save_path)))

    def test_139_graph_no_data(self):
        """Catch the case where no data is passed to GraphHisto"""
        input_array = Vector()
        self.assertRaises(NoDataError, lambda: GraphHisto(input_array))

    def test_140_graph_vector(self):
        """Generate a histogram from a Vector object"""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=5000))
        self.assertTrue(GraphHisto(input_array, save_to='{}test_histo_140'.format(self.save_path)))

    def test_141_graph_groups(self):
        """Generate a histogram from a Vector with groups"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=2500)
        grp1 = ['one'] * 2500
        grp2 = ['two'] * 2500
        exp = Vector(input_array, groups=grp1).append(Vector(input_array, groups=grp2))
        self.assertTrue(GraphHisto(exp, save_to='{}test_histo_141'.format(self.save_path)))


if __name__ == '__main__':
    unittest.main()
