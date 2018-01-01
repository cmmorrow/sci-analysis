import unittest
import numpy as np
import pandas as pd
import scipy.stats as st
from ..analysis import determine_analysis_type
from ..analysis.exc import NoDataError
from ..data import Vector, Categorical


class MyTestCase(unittest.TestCase):

    def test_small_float_array(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30)
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float_list(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30).tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(st.norm.rvs(0, 1, 30))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_float_array(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 10000)
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_float_list(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 10000).tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_float_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(st.norm.rvs(0, 1, 10000))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float32_array(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30).astype('float32')
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float32_list(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30).astype('float32').tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float32_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(st.norm.rvs(0, 1, 30).astype('float32'))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float16_array(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30).astype('float16')
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float16_list(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30).astype('float16').tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_float16_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(st.norm.rvs(0, 1, 30).astype('float16'))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_single_float_array(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 1)
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_single_float_list(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 1).tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_single_float_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(st.norm.rvs(0, 1, 1))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_vector(self):
        np.random.seed(123456789)
        input_array = Vector(st.norm.rvs(0, 1, 30))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_vector(self):
        np.random.seed(123456789)
        input_array = Vector(st.norm.rvs(0, 1, 10000))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_array_with_nan(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30)
        input_array[4] = np.nan
        input_array[10] = np.nan
        input_array[17] = np.nan
        input_array[22] = np.nan
        input_array[24] = np.nan
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_list_with_nan(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30)
        input_array[4] = np.nan
        input_array[10] = np.nan
        input_array[17] = np.nan
        input_array[22] = np.nan
        input_array[24] = np.nan
        self.assertIsInstance(determine_analysis_type(input_array.tolist()), Vector)

    def test_small_series_with_nan(self):
        np.random.seed(123456789)
        input_array = st.norm.rvs(0, 1, 30)
        input_array[4] = np.nan
        input_array[10] = np.nan
        input_array[17] = np.nan
        input_array[22] = np.nan
        input_array[24] = np.nan
        self.assertIsInstance(determine_analysis_type(pd.Series(input_array)), Vector)

    def test_none(self):
        input_array = None
        self.assertRaises(ValueError, lambda: determine_analysis_type(input_array))

    def test_empty_list(self):
        input_array = list()
        self.assertRaises(NoDataError, lambda: determine_analysis_type(input_array))

    def test_empty_array(self):
        input_array = np.array([])
        self.assertRaises(NoDataError, lambda: determine_analysis_type(input_array))

    def test_empty_vector(self):
        input_array = Vector([])
        self.assertRaises(NoDataError, lambda: determine_analysis_type(input_array))

    def test_float_scalar(self):
        input_array = 3.14159256
        self.assertRaises(ValueError, lambda: determine_analysis_type(input_array))

    def test_small_int_array(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30)
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int_list(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(np.random.randint(-10, 11, 30))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_int_array(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 10000)
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_int_list(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 10000).tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_int_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(np.random.randint(-10, 11, 10000))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int32_array(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).astype('int32')
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int32_list(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).astype('int32').tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int32_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(np.random.randint(-10, 11, 30).astype('int32'))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int16_array(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).astype('int16')
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int16_list(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).astype('int16').tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int16_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(np.random.randint(-10, 11, 30).astype('int16'))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int8_array(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).astype('int8')
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int8_list(self):
        np.random.seed(123456789)
        input_array = np.random.randint(-10, 11, 30).astype('int8').tolist()
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_int8_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(np.random.randint(-10, 11, 30).astype('int8'))
        self.assertIsInstance(determine_analysis_type(input_array), Vector)

    def test_int_scalar(self):
        input_array = 3
        self.assertRaises(ValueError, lambda: determine_analysis_type(input_array))

    def test_small_cat_list(self):
        np.random.seed(123456789)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_cat_array(self):
        np.random.seed(123456789)
        input_array = np.array(['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_cat_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_cat_list(self):
        np.random.seed(123456789)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(10000)]
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_cat_array(self):
        np.random.seed(123456789)
        input_array = np.array(['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(10000)])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_large_cat_series(self):
        np.random.seed(123456789)
        input_array = pd.Series(['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(10000)])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_single_cat_list(self):
        input_array = ['a']
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_single_cat_array(self):
        input_array = np.array(['a'])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_single_cat_series(self):
        input_array = pd.Series(['a'])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_categorical(self):
        np.random.seed(123456789)
        input_array = Categorical(['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_large_categorical(self):
        np.random.seed(123456789)
        input_array = Categorical(['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(10000)])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_string_scalar(self):
        input_array = 'a'
        self.assertRaises(ValueError, lambda: determine_analysis_type(input_array))

    def test_empty_categorical(self):
        input_array = Categorical([])
        self.assertRaises(NoDataError, lambda: determine_analysis_type(input_array))

    def test_small_cat_list_with_nan(self):
        np.random.seed(123456789)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        input_array[4] = np.nan
        input_array[10] = np.nan
        input_array[17] = np.nan
        input_array[22] = np.nan
        input_array[24] = np.nan
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_cat_array_with_nan(self):
        np.random.seed(123456789)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        input_array[4] = np.nan
        input_array[10] = np.nan
        input_array[17] = np.nan
        input_array[22] = np.nan
        input_array[24] = np.nan
        input_array = np.array(input_array)
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_cat_series_with_nan(self):
        np.random.seed(123456789)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        input_array[4] = np.nan
        input_array[10] = np.nan
        input_array[17] = np.nan
        input_array[22] = np.nan
        input_array[24] = np.nan
        input_array = pd.Series(input_array)
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)
        self.assertNotIsInstance(determine_analysis_type(input_array), Vector)

    def test_small_string_num_list(self):
        input_array = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_small_string_num_array(self):
        input_array = np.array(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_small_string_num_series(self):
        input_array = pd.Series(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_small_mixed_list(self):
        input_array = ['1', 'a', np.nan, 4, 5.0]
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_small_mixed_array(self):
        input_array = np.array(['1', 'a', np.nan, 4, 5.0])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_small_mixed_series(self):
        input_array = pd.Series(['1', 'a', np.nan, 4, 5.0])
        self.assertIsInstance(determine_analysis_type(input_array), Categorical)

    def test_arrays_with_other(self):
        np.random.seed(123456789)
        input_1_array = st.norm.rvs(0, 1, 10000)
        input_2_array = st.norm.rvs(1, 1, 10000)
        self.assertIsInstance(determine_analysis_type(input_1_array, other=input_2_array), Vector)
        self.assertTrue(pd.Series(input_1_array)
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).data))
        self.assertTrue(pd.Series(input_2_array)
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).other))

    def test_series_with_other(self):
        np.random.seed(123456789)
        input_1_array = pd.Series(st.norm.rvs(0, 1, 10000))
        input_2_array = pd.Series(st.norm.rvs(1, 1, 10000))
        self.assertIsInstance(determine_analysis_type(input_1_array, other=input_2_array), Vector)
        self.assertTrue(input_1_array
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).data))
        self.assertTrue(input_2_array
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).other))

    def test_list_with_other(self):
        np.random.seed(123456789)
        input_1_array = pd.Series(st.norm.rvs(0, 1, 10000)).tolist()
        input_2_array = pd.Series(st.norm.rvs(1, 1, 10000)).tolist()
        self.assertIsInstance(determine_analysis_type(input_1_array, other=input_2_array), Vector)
        self.assertListEqual(input_1_array, determine_analysis_type(input_1_array, other=input_2_array).data.tolist())
        self.assertListEqual(input_2_array, determine_analysis_type(input_1_array, other=input_2_array).other.tolist())

    def test_vector_with_other(self):
        np.random.seed(123456789)
        input_1_array = st.norm.rvs(0, 1, 10000)
        input_2_array = st.norm.rvs(1, 1, 10000)
        vector = Vector(input_1_array, other=input_2_array)
        self.assertIsInstance(determine_analysis_type(vector), Vector)
        self.assertTrue(vector.data
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).data))
        self.assertTrue(vector.other
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).other))

    def test_vector_with_other_categorical(self):
        np.random.seed(123456789)
        input_1_array = st.norm.rvs(0, 1, 10000)
        input_2_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        self.assertIsInstance(determine_analysis_type(input_1_array, other=input_2_array), Vector)
        self.assertTrue(pd.Series(input_1_array)
                        .equals(determine_analysis_type(input_1_array, other=input_2_array).data))
        self.assertTrue(all(determine_analysis_type(input_1_array, other=input_2_array).other.isnull()))

    def test_categorical_with_other_vector(self):
        np.random.seed(123456789)
        input_1_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        input_2_array = st.norm.rvs(0, 1, 10000)
        self.assertIsInstance(determine_analysis_type(input_1_array, other=input_2_array), Categorical)

    def test_float_with_groups(self):
        np.random.seed(123456789)
        input_1_array = pd.DataFrame({'input': st.norm.rvs(size=2000), 'group': ['Group 1'] * 2000})
        input_2_array = pd.DataFrame({'input': st.norm.rvs(1, size=2000), 'group': ['Group 2'] * 2000})
        df = pd.concat([input_1_array, input_2_array])
        self.assertIsInstance(determine_analysis_type(df['input'], groups=df['group']), Vector)
        self.assertEqual(len(determine_analysis_type(df['input'], groups=df['group']).groups), 2)

    def test_float_with_other_with_groups(self):
        np.random.seed(123456789)
        input_1_array = pd.DataFrame({'input1': st.norm.rvs(size=2000),
                                      'input2': st.weibull_min.rvs(1.7, size=2000),
                                      'group': ['Group 1'] * 2000})
        input_2_array = pd.DataFrame({'input1': st.norm.rvs(1, size=2000),
                                      'input2': st.weibull_min.rvs(1.7, size=2000),
                                      'group': ['Group 2'] * 2000})
        df = pd.concat([input_1_array, input_2_array])
        self.assertIsInstance(determine_analysis_type(df['input1'], other=df['input2'], groups=df['group']), Vector)
        self.assertEqual(len(determine_analysis_type(df['input1'], other=df['input2'], groups=df['group']).groups), 2)

    def test_categorical_with_groups(self):
        np.random.seed(123456789)
        input_array = ['abcdefghijklmnopqrstuvwxyz'[:np.random.randint(1, 26)] for _ in range(30)]
        grp = ['Group 1' for _ in range(30)]
        self.assertIsInstance(determine_analysis_type(input_array, groups=grp), Categorical)


if __name__ == '__main__':
    unittest.main()
