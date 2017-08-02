import unittest
from numpy import nan
from numpy.random import seed, randint
from pandas import Series

from analysis import CategoricalStatistics, NoDataError
from data import Categorical, NumberOfCategoriesWarning
from test.test_categorical import TestWarnings


class MyTestCase(TestWarnings):

    def test_100_categorical_stats_simple_unordered(self):
        input_array = ['one', 'two', 'one', 'three']
        obj = CategoricalStatistics(input_array, display=False)
        output = """

Statistics
----------

Rank          Frequency     Category      
------------------------------------------
1             2             one           
2             1             three         
3             1             two           """
        self.assertEqual(str(obj), output)
        self.assertEqual(obj.name, 'Statistics')
        self.assertTrue(obj.data.data.equals(Series(input_array).astype('category')))
        self.assertListEqual(obj.results, [{'Rank': 1, 'Category': 'one', 'Frequency': 2},
                                           {'Rank': 2, 'Category': 'three', 'Frequency': 1},
                                           {'Rank': 3, 'Category': 'two', 'Frequency': 1}])

    def test_101_categorical_stats_simple_ordered_categories(self):
        input_array = Categorical(['one', 'two', 'one', 'three'], order=['three', 'two', 'one'])
        obj = CategoricalStatistics(input_array, display=False)
        output = """

Statistics
----------

Rank          Frequency     Category      
------------------------------------------
2             1             three         
3             1             two           
1             2             one           """
        self.assertEqual(str(obj), output)
        self.assertListEqual(obj.results, [{'Frequency': 1, 'Category': 'three', 'Rank': 2},
                                           {'Frequency': 1, 'Category': 'two', 'Rank': 3},
                                           {'Frequency': 2, 'Category': 'one', 'Rank': 1}])

    def test_102_categorical_stats_with_na(self):
        seed(987654321)
        src = 'abcdefghijklmnop'
        input_array = [src[:randint(1, 17)] for _ in range(50)]
        input_array[3] = nan
        input_array[14] = nan
        input_array[22] = nan
        input_array[28] = nan
        output = """

Statistics
----------

Rank          Frequency     Category      
------------------------------------------
1             6             abcdefghijklmnop
2             5             abc           
3             5             abcdefg       
4             5             abcdefghijk   
5             4             abcdefgh      
6             4              nan          
7             3             a             
8             3             ab            
9             3             abcdefghij    
10            3             abcdefghijklm 
11            2             abcde         
12            2             abcdefghi     
13            2             abcdefghijklmno
14            1             abcd          
15            1             abcdef        
16            1             abcdefghijkl  """
        test = CategoricalStatistics(input_array, display=False)
        self.assertTrue(str(test), output)

    def test_103_no_data(self):
        input_array = None
        self.assertRaises(NoDataError, lambda: CategoricalStatistics(input_array, display=False))
        input_array = Categorical(['a', 'b', 'a', 'c', 'c', 'd'], order=['z', 'y', 'x', 'w'], dropna=True)
        self.assertRaises(NoDataError, lambda: CategoricalStatistics(input_array, display=False))

    def test_104_no_data_except_nan(self):
        input_array = Categorical(['a', 'b', 'a', 'c', 'c', 'd'], order=['z', 'y', 'x', 'w'])
        output = """

Statistics
----------

Rank          Frequency     Category      
------------------------------------------
2             0             z             
3             0             y             
4             0             x             
5             0             w             
1             6              nan          """
        test = CategoricalStatistics(input_array, display=False)
        self.assertTrue(str(test), output)

    def test_105_too_many_categories_warning(self):
        input_array = [str(x) for x in range(100)]
        self.assertWarnsCrossCompatible(NumberOfCategoriesWarning, lambda: CategoricalStatistics(input_array, False))


if __name__ == '__main__':
    unittest.main()
