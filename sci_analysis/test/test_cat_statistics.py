import unittest
from numpy import nan
from numpy.random import seed, randint
from pandas import Series

from ..analysis import CategoricalStatistics
from ..analysis.exc import NoDataError
from ..data import Categorical, NumberOfCategoriesWarning
from .test_categorical import TestWarnings


class MyTestCase(TestWarnings):

    def test_100_categorical_stats_simple_unordered(self):
        input_array = ['one', 'two', 'one', 'three']
        obj = CategoricalStatistics(input_array, display=False)
        output = """

Overall Statistics
------------------

Total            =  4
Number of Groups =  3


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             2              50.0000      one           
2             1              25.0000      three         
2             1              25.0000      two           """
        self.assertEqual(str(obj), output)
        self.assertEqual(obj.name, 'Statistics')
        self.assertTrue(obj.data.data.equals(Series(input_array).astype('category')))
        self.assertDictEqual(obj.results[0], {'Total': 4, 'Number of Groups': 3})
        self.assertListEqual(obj.results[1], [{'Rank': 1, 'Category': 'one', 'Frequency': 2, 'Percent': 50.0},
                                              {'Rank': 2, 'Category': 'three', 'Frequency': 1, 'Percent': 25.0},
                                              {'Rank': 2, 'Category': 'two', 'Frequency': 1, 'Percent': 25.0}])

    def test_101_categorical_stats_simple_ordered_categories(self):
        input_array = ['one', 'two', 'one', 'three']
        obj = CategoricalStatistics(input_array, order=['three', 'two', 'one'], display=False)
        output = """

Overall Statistics
------------------

Total            =  4
Number of Groups =  3


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
2             1              25.0000      three         
2             1              25.0000      two           
1             2              50.0000      one           """
        self.assertEqual(str(obj), output)
        self.assertDictEqual(obj.results[0], {'Total': 4, 'Number of Groups': 3})
        self.assertListEqual(obj.results[1], [{'Frequency': 1, 'Category': 'three', 'Rank': 2, 'Percent': 25},
                                              {'Frequency': 1, 'Category': 'two', 'Rank': 2, 'Percent': 25},
                                              {'Frequency': 2, 'Category': 'one', 'Rank': 1, 'Percent': 50}])

    def test_102_categorical_stats_with_na(self):
        seed(987654321)
        src = 'abcdefghijklmnop'
        input_array = [src[:randint(1, 17)] for _ in range(50)]
        input_array[3] = nan
        input_array[14] = nan
        input_array[22] = nan
        input_array[28] = nan
        output = """

Overall Statistics
------------------

Total            =  50
Number of Groups =  16


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             6              12.0000      abcdefghijklmnop
2             5              10.0000      abc           
2             5              10.0000      abcdefg       
2             5              10.0000      abcdefghijk   
3             4              8.0000       abcdefgh      
3             4              8.0000        nan          
4             3              6.0000       a             
4             3              6.0000       ab            
4             3              6.0000       abcdefghij    
4             3              6.0000       abcdefghijklm 
5             2              4.0000       abcde         
5             2              4.0000       abcdefghi     
5             2              4.0000       abcdefghijklmno
6             1              2.0000       abcd          
6             1              2.0000       abcdef        
6             1              2.0000       abcdefghijkl  """
        test = CategoricalStatistics(input_array, display=False)
        self.assertEqual(str(test), output)
        self.assertDictEqual(test.results[0], {'Total': 50, 'Number of Groups': 16})
        self.assertListEqual(test.results[1], [{'Frequency': 6, 'Category': 'abcdefghijklmnop', 'Rank': 1, 'Percent': 12.},
                                               {'Frequency': 5, 'Category': 'abc', 'Rank': 2, 'Percent': 10.0},
                                               {'Frequency': 5, 'Category': 'abcdefg', 'Rank': 2, 'Percent': 10.0},
                                               {'Frequency': 5, 'Category': 'abcdefghijk', 'Rank': 2, 'Percent': 10.0},
                                               {'Frequency': 4, 'Category': 'abcdefgh', 'Rank': 3, 'Percent': 8.0},
                                               {'Frequency': 4, 'Category': nan, 'Rank': 3, 'Percent': 8.0},
                                               {'Frequency': 3, 'Category': 'a', 'Rank': 4, 'Percent': 6.0},
                                               {'Frequency': 3, 'Category': 'ab', 'Rank': 4, 'Percent': 6.0},
                                               {'Frequency': 3, 'Category': 'abcdefghij', 'Rank': 4, 'Percent': 6.0},
                                               {'Frequency': 3, 'Category': 'abcdefghijklm', 'Rank': 4, 'Percent': 6.0},
                                               {'Frequency': 2, 'Category': 'abcde', 'Rank': 5, 'Percent': 4.0},
                                               {'Frequency': 2, 'Category': 'abcdefghi', 'Rank': 5, 'Percent': 4.0},
                                               {'Frequency': 2, 'Category': 'abcdefghijklmno', 'Rank': 5, 'Percent': 4.0},
                                               {'Frequency': 1, 'Category': 'abcd', 'Rank': 6, 'Percent': 2.0},
                                               {'Frequency': 1, 'Category': 'abcdef', 'Rank': 6, 'Percent': 2.0},
                                               {'Frequency': 1, 'Category': 'abcdefghijkl', 'Rank': 6, 'Percent': 2.0}])

    def test_103_no_data(self):
        input_array = None
        self.assertRaises(NoDataError, lambda: CategoricalStatistics(input_array, display=False))
        input_array = Categorical(['a', 'b', 'a', 'c', 'c', 'd'], order=['z', 'y', 'x', 'w'], dropna=True)
        self.assertRaises(NoDataError, lambda: CategoricalStatistics(input_array, display=False))
        input_array = []
        self.assertRaises(NoDataError, lambda: CategoricalStatistics(input_array, display=False))

    def test_104_no_data_except_nan(self):
        input_array = ['a', 'b', 'a', 'c', 'c', 'd']
        output = """

Overall Statistics
------------------

Total            =  6
Number of Groups =  5


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             6              100.0000      nan          
2             0              0.0000       z             
2             0              0.0000       y             
2             0              0.0000       x             
2             0              0.0000       w             """
        test = CategoricalStatistics(input_array, order=['z', 'y', 'x', 'w'], display=False)
        self.assertEqual(str(test), output)
        self.assertDictEqual(test.results[0], {'Total': 6, 'Number of Groups': 5})
        self.assertListEqual(test.results[1], [{'Rank': 1, 'Frequency': 6, 'Percent': 100, 'Category': nan},
                                               {'Rank': 2, 'Frequency': 0, 'Percent': 0, 'Category': 'z'},
                                               {'Rank': 2, 'Frequency': 0, 'Percent': 0, 'Category': 'y'},
                                               {'Rank': 2, 'Frequency': 0, 'Percent': 0, 'Category': 'x'},
                                               {'Rank': 2, 'Frequency': 0, 'Percent': 0, 'Category': 'w'}])

    def test_105_too_many_categories_warning(self):
        input_array = [str(x) for x in range(100)]
        self.assertWarnsCrossCompatible(NumberOfCategoriesWarning,
                                        lambda: CategoricalStatistics(input_array, display=False))

    def test_106_single_category(self):
        input_array = ['a', 'b', 'a', 'c', 'c', 'd']
        order = []
        output = """

Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             6              100.0000      nan          """
        test = CategoricalStatistics(input_array, order=order, display=False)
        self.assertEqual(str(test), output)
        self.assertListEqual(test.results, [{'Frequency': 6, 'Category': nan, 'Rank': 1, 'Percent': 100.0}])
        order = 'c'
        output = """

Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             2              100.0000     c             """
        test = CategoricalStatistics(input_array, order=order, display=False, dropna=True)
        self.assertEqual(str(test), output)
        self.assertListEqual(test.results, [{'Frequency': 2, 'Category': 'c', 'Rank': 1, 'Percent': 100.0}])

    def test_107_numeric_group_name(self):
        input_array = [1., 2., 1., 3., 3., 4.]
        output = """

Overall Statistics
------------------

Total            =  6
Number of Groups =  4


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             2              33.3333      1             
1             2              33.3333      3             
2             1              16.6667      2             
2             1              16.6667      4             """
        exp = CategoricalStatistics(input_array, display=False)
        self.assertEqual(str(exp), output)

    def test_108_year_group_name(self):
        input_array = [2015, 2016, 2017, 2018, 2019]
        exp = CategoricalStatistics(input_array, display=False)
        output = """

Overall Statistics
------------------

Total            =  5
Number of Groups =  5


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             1              20.0000      2015          
1             1              20.0000      2016          
1             1              20.0000      2017          
1             1              20.0000      2018          
1             1              20.0000      2019          """
        self.assertEqual(str(exp), output)

    def test_109_float_group_name(self):
        input_array = [.123, .456, .789]
        exp = CategoricalStatistics(input_array, display=True)
        output = """

Overall Statistics
------------------

Total            =  3
Number of Groups =  3


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1             1              33.3333       0.1230       
1             1              33.3333       0.4560       
1             1              33.3333       0.7890       """
        self.assertEqual(str(exp), output)


if __name__ == '__main__':
    unittest.main()
