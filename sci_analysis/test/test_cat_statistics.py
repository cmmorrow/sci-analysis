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

Total                =  4
Number of Categories =  3


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
        self.assertDictEqual(obj.results[0], {'Total': 4, 'Number of Categories': 3})
        self.assertListEqual(obj.results[1], [{'Rank': 1, 'Category': 'one', 'Frequency': 2, 'Percent': 50.0},
                                              {'Rank': 2, 'Category': 'three', 'Frequency': 1, 'Percent': 25.0},
                                              {'Rank': 2, 'Category': 'two', 'Frequency': 1, 'Percent': 25.0}])

    def test_101_categorical_stats_simple_ordered_categories(self):
        input_array = ['one', 'two', 'one', 'three']
        obj = CategoricalStatistics(input_array, order=['three', 'two', 'one'], display=False)
        output = """

Overall Statistics
------------------

Total                =  4
Number of Categories =  3


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
2             1              25.0000      three         
2             1              25.0000      two           
1             2              50.0000      one           """
        self.assertEqual(str(obj), output)
        self.assertDictEqual(obj.results[0], {'Total': 4, 'Number of Categories': 3})
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

Total                =  50
Number of Categories =  16


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
        self.assertDictEqual(test.results[0], {'Total': 50, 'Number of Categories': 16})
        self.assertListEqual(test.results[1], [
            {'Frequency': 6, 'Category': 'abcdefghijklmnop', 'Rank': 1, 'Percent': 12.},
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
            {'Frequency': 1, 'Category': 'abcdefghijkl', 'Rank': 6, 'Percent': 2.0}
        ])

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

Total                =  6
Number of Categories =  5


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
        self.assertDictEqual(test.results[0], {'Total': 6, 'Number of Categories': 5})
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

    def test_107_categorical_stats_simple_with_groups(self):
        input_array = ['one', 'two', 'one', 'three']
        groups = ['a', 'b', 'b', 'a']
        obj = CategoricalStatistics(input_array, groups=groups, display=False)
        output = """

Overall Statistics
------------------

Total                =  4
Number of Categories =  6


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1              1.0000        25.0000      one, a        
1              1.0000        25.0000      one, b        
1              1.0000        25.0000      three, a      
1              1.0000        25.0000      two, b        
2              0.0000        0.0000       three, b      
2              0.0000        0.0000       two, a        """
        self.assertEqual(str(obj), output)
        self.assertEqual(obj.name, 'Statistics')
        self.assertTrue(obj.data.data.equals(Series(input_array, name='ind').astype('category')))
        self.assertDictEqual(obj.results[0], {'Total': 4, 'Number of Categories': 6})
        self.assertListEqual(obj.results[1], [
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'one, a'},
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'one, b'},
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'three, a'},
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'two, b'},
            {'Frequency': 0.0, 'Rank': 2, 'Percent': 0.0, 'Category': 'three, b'},
            {'Frequency': 0.0, 'Rank': 2, 'Percent': 0.0, 'Category': 'two, a'},
        ])

    def test_108_categorical_stats_simple_with_groups_ordered(self):
        input_array = ['one', 'two', 'one', 'three']
        groups = ['a', 'b', 'b', 'a']
        order = ['three', 'two', 'one']
        obj = CategoricalStatistics(input_array, order=order, groups=groups, display=False)
        output = """

Overall Statistics
------------------

Total                =  4
Number of Categories =  6


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1              1.0000        25.0000      three, a      
2              0.0000        0.0000       three, b      
2              0.0000        0.0000       two, a        
1              1.0000        25.0000      two, b        
1              1.0000        25.0000      one, a        
1              1.0000        25.0000      one, b        """
        self.assertEqual(str(obj), output)
        self.assertEqual(obj.name, 'Statistics')
        self.assertTrue(obj.data.data.equals(Series(input_array, name='ind')
                                             .astype('category')
                                             .cat.reorder_categories(order, ordered=True)))
        self.assertDictEqual(obj.results[0], {'Total': 4, 'Number of Categories': 6})
        self.assertListEqual(obj.results[1], [
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'three, a'},
            {'Frequency': 0.0, 'Rank': 2, 'Percent': 0.0, 'Category': 'three, b'},
            {'Frequency': 0.0, 'Rank': 2, 'Percent': 0.0, 'Category': 'two, a'},
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'two, b'},
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'one, a'},
            {'Frequency': 1.0, 'Rank': 1, 'Percent': 25.0, 'Category': 'one, b'},
        ])

    def test_109_categorical_stats_with_na_and_groups(self):
        seed(987654321)
        src = 'abcdefghijklmnop'
        input_array = [src[:randint(1, 17)] for _ in range(20)]
        input_array[3] = nan
        input_array[11] = nan
        input_array[14] = nan
        input_array[17] = nan
        groups = (1, 2, 3, 4, 5) * 4
        output = """

Overall Statistics
------------------

Total                =  20
Number of Categories =  50


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1              2.0000        12.5000      abcdefgh, 3   
2              1.0000        6.2500       abcdefghij, 4 
2              1.0000        6.2500       abc, 2        
2              1.0000        6.2500       abcdefg, 4    
2              1.0000        6.2500       abcdefghijklmnop, 4
2              1.0000        6.2500       abcdefghij, 2 
2              1.0000        6.2500       ab, 3         
2              1.0000        6.2500       abcdefgh, 1   
2              1.0000        6.2500       ab, 1         
2              1.0000        6.2500       a, 5          
2              1.0000        6.2500       abcdefghijklmnop, 1
2              1.0000        6.2500       abcdefghijklmnop, 2
2              1.0000        6.2500       abcdefghi, 1  
2              1.0000        6.2500       abcdefghijklmno, 5
2              1.0000        6.2500       abcd, 5       
3              0.0000        0.0000       abcdefghi, 2  
3              0.0000        0.0000       abcdefghi, 3  
3              0.0000        0.0000       abcdefghi, 4  
3              0.0000        0.0000       a, 1          
3              0.0000        0.0000       abcdefghij, 1 
3              0.0000        0.0000       abcdefghij, 3 
3              0.0000        0.0000       abcdefghij, 5 
3              0.0000        0.0000       abcdefghijklmno, 1
3              0.0000        0.0000       abcdefghijklmno, 2
3              0.0000        0.0000       abcdefghijklmno, 3
3              0.0000        0.0000       abcdefghijklmno, 4
3              0.0000        0.0000       abcdefghijklmnop, 3
3              0.0000        0.0000       abcdefghi, 5  
3              0.0000        0.0000       abcdefgh, 5   
3              0.0000        0.0000       abcdefg, 5    
3              0.0000        0.0000       abcdefgh, 2   
3              0.0000        0.0000       a, 2          
3              0.0000        0.0000       a, 3          
3              0.0000        0.0000       a, 4          
3              0.0000        0.0000       ab, 2         
3              0.0000        0.0000       ab, 4         
3              0.0000        0.0000       ab, 5         
3              0.0000        0.0000       abc, 1        
3              0.0000        0.0000       abc, 3        
3              0.0000        0.0000       abc, 4        
3              0.0000        0.0000       abc, 5        
3              0.0000        0.0000       abcd, 1       
3              0.0000        0.0000       abcd, 2       
3              0.0000        0.0000       abcd, 3       
3              0.0000        0.0000       abcd, 4       
3              0.0000        0.0000       abcdefg, 1    
3              0.0000        0.0000       abcdefg, 2    
3              0.0000        0.0000       abcdefg, 3    
3              0.0000        0.0000       abcdefgh, 4   
3              0.0000        0.0000       abcdefghijklmnop, 5"""
        test = CategoricalStatistics(input_array, groups=groups, display=False)
        self.assertEqual(str(test), output)
        self.assertDictEqual(test.results[0], {'Total': 20, 'Number of Categories': 50})
        self.assertListEqual(test.results[1], [
            {'Frequency': 2.0, 'Rank': 1, 'Percent': 12.5, 'Category': 'abcdefgh, 3'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghij, 4'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abc, 2'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefg, 4'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghijklmnop, 4'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghij, 2'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'ab, 3'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefgh, 1'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'ab, 1'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'a, 5'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghijklmnop, 1'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghijklmnop, 2'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghi, 1'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcdefghijklmno, 5'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 6.25, 'Category': 'abcd, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghi, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghi, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghi, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'a, 1'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghij, 1'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghij, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghij, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghijklmno, 1'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghijklmno, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghijklmno, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghijklmno, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghijklmnop, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghi, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefgh, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefg, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefgh, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'a, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'a, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'a, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'ab, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'ab, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'ab, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abc, 1'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abc, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abc, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abc, 5'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcd, 1'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcd, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcd, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcd, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefg, 1'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefg, 2'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefg, 3'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefgh, 4'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'abcdefghijklmnop, 5'},
        ])

    def test_110_categorical_stats_with_nan_group(self):
        input_array = ['a', 'b', 'a', 'c', 'c', 'd']
        groups = [1, nan, 1, nan, 1, nan]
        output = """

Overall Statistics
------------------

Total                =  6
Number of Categories =  4


Statistics
----------

Rank          Frequency     Percent       Category      
--------------------------------------------------------
1              2.0000        66.6667      a, 1.0        
2              1.0000        33.3333      c, 1.0        
3              0.0000        0.0000       b, 1.0        
3              0.0000        0.0000       d, 1.0        """
        test = CategoricalStatistics(input_array, groups=groups, display=False)
        self.assertEqual(str(test), output)
        self.assertDictEqual(test.results[0], {'Total': 6, 'Number of Categories': 4})
        self.assertListEqual(test.results[1], [
            {'Frequency': 2.0, 'Rank': 1, 'Percent': 66.66666666666666, 'Category': 'a, 1.0'},
            {'Frequency': 1.0, 'Rank': 2, 'Percent': 33.33333333333333, 'Category': 'c, 1.0'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'b, 1.0'},
            {'Frequency': 0.0, 'Rank': 3, 'Percent': 0.0, 'Category': 'd, 1.0'},
        ])


if __name__ == '__main__':
    unittest.main()
