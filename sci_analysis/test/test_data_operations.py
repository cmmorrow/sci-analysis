import unittest

import numpy as np
import numpy.ma as ma
import pandas as pd

from ..data import (is_array, is_dict, is_dict_group, is_group, is_iterable, is_tuple, to_float, flatten, is_series,
                    Vector, is_data, is_vector, is_numeric, is_number)


class MyTestCase(unittest.TestCase):
    inputs = {
        'num': 3,
        'float': 1.34,
        'string': "hello",
        'num_string': '1.34',
        'char': "h",
        'none': None,
        'list': [1, 2, 3, 4, 5],
        'num_list': ["1", "2", "3", "4", "5"],
        'mixed_list': [1, 2.00, "3", "four", '5'],
        'zero_len_list': [],
        'multiple_dim_list': [[1, 2, 3], [4, 5, 6]],
        'tuple': (1, 2, 3, 4, 5),
        'num_tuple': ("1", "2", "3", "4", "5"),
        'mixed_tuple': (1, 2, "3", "four", '5'),
        'dict': {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
        'array': np.array([1, 2, 3, 4, 5]),
        'float_array': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'nan_array': np.array([1, float("nan"), 3, float("nan"), 5], dtype='float'),
        'negative_array': np.array([-1, 2.0, -3.00, 0, -5]),
        'masked_array': ma.masked_array([1, 2, 3, 4, 5], mask=[0, 1, 1, 0, 0]),
        'multi_dim_array': np.array([[1, 2, 3], [4, 5, 6]]),
        'scalar_array': np.array(3),
        'zero_len_array': np.array([]),
        'empty_array': np.empty(5),
        'vector': Vector([1, 2, 3, 4, 5]),
        'series': pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]),
        'dict_series': pd.Series({1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0, 5: 5.0}),
        'large_array': np.random.rand(500),
        'large_list': range(500),
        'group': [np.random.rand(50), np.random.rand(50) * 2, np.random.rand(50) * 3],
        'group_of_lists': [range(5), range(6, 10), range(11, 15)],
        'dict_of_lists': {'a': range(1, 5), 'b': range(6, 10), 'c': range(11, 15)}
    }

    ans_array = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 1,
        'float_array': 1,
        'nan_array': 1,
        'negative_array': 1,
        'masked_array': 1,
        'multi_dim_array': 1,
        'scalar_array': 1,
        'zero_len_array': 1,
        'empty_array': 1,
        'vector': 0,
        'series': 1,
        'dict_series': 1,
        'large_array': 1,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    ans_dict = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 1,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 0,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 1
    }

    ans_iterable = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 1,
        'num_list': 1,
        'mixed_list': 1,
        'zero_len_list': 1,
        'multiple_dim_list': 1,
        'tuple': 1,
        'num_tuple': 1,
        'mixed_tuple': 1,
        'dict': 1,
        'array': 1,
        'float_array': 1,
        'nan_array': 1,
        'negative_array': 1,
        'masked_array': 1,
        'multi_dim_array': 1,
        'scalar_array': 0,
        'zero_len_array': 1,
        'empty_array': 1,
        'vector': 1,
        'series': 1,
        'dict_series': 1,
        'large_array': 1,
        'large_list': 1,
        'group': 1,
        'group_of_lists': 1,
        'dict_of_lists': 1
    }

    ans_tuple = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 1,
        'num_tuple': 1,
        'mixed_tuple': 1,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 0,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    ans_data = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 1,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    ans_vector = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 1,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    ans_group = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 1,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 1,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 0,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 1,
        'group_of_lists': 1,
        'dict_of_lists': 0
    }

    ans_dict_group = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 0,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 1
    }

    ans_series = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 0,
        'series': 1,
        'dict_series': 1,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    ans_numeric = {
        'num': 0,
        'float': 0,
        'num_string': 0,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 0,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 1,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    ans_number = {
        'num': 1,
        'float': 1,
        'num_string': 1,
        'string': 0,
        'char': 0,
        'none': 0,
        'list': 0,
        'num_list': 0,
        'mixed_list': 0,
        'zero_len_list': 0,
        'multiple_dim_list': 0,
        'tuple': 0,
        'num_tuple': 0,
        'mixed_tuple': 0,
        'dict': 0,
        'array': 0,
        'float_array': 0,
        'nan_array': 0,
        'negative_array': 0,
        'masked_array': 0,
        'multi_dim_array': 0,
        'scalar_array': 1,
        'zero_len_array': 0,
        'empty_array': 0,
        'vector': 0,
        'series': 0,
        'dict_series': 0,
        'large_array': 0,
        'large_list': 0,
        'group': 0,
        'group_of_lists': 0,
        'dict_of_lists': 0
    }

    # Test logic tests

    def test_001_is_array(self):
        """Tests the is_array method"""
        eval_array = {}
        print("")
        print("is_array test")
        print("-" * 80)
        for name, test in self.inputs.items():
            try:
                assert is_array(test)
                print("PASS: " + name)
                eval_array[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_array[name] = 0
        # self.assertTrue(eval_array == self.ans_array, "FAIL: is_array test")
        self.assertDictEqual(eval_array, self.ans_array, "FAIL: is_array test")

    def test_002_is_dict(self):
        """Tests the is_dict method"""
        eval_dict = {}
        print("")
        print("is_dict test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_dict(test)
                print("PASS: " + name)
                eval_dict[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_dict[name] = 0
        self.assertTrue(eval_dict == self.ans_dict, "FAIL: is_dict test")

    def test_003_is_iterable(self):
        """Tests the is_iterable method"""
        eval_iterable = {}
        print("")
        print("is_iterable test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_iterable(test)
                print("PASS: " + name)
                eval_iterable[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_iterable[name] = 0
        self.assertTrue(eval_iterable == self.ans_iterable, "FAIL: is_iterable test")

    def test_004_is_tuple(self):
        """Tests the is_tuple method"""
        eval_tuple = {}
        print("")
        print("is_tuple test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_tuple(test)
                print("PASS: " + name)
                eval_tuple[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_tuple[name] = 0
        self.assertTrue(eval_tuple == self.ans_tuple, "FAIL: is_tuple test")

    def test_005_is_data(self):
        """Tests the is_data method"""
        eval_data = {}
        print("")
        print("is_data test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_data(test)
                print("PASS: " + name)
                eval_data[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_data[name] = 0
        self.assertTrue(eval_data == self.ans_data, "FAIL: is_data test")

    def test_006_is_vector(self):
        """Tests the is_vector method"""
        eval_vector = {}
        print("")
        print("is_vector test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_vector(test)
                print("PASS: " + name)
                eval_vector[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_vector[name] = 0
        self.assertTrue(eval_vector == self.ans_vector, "FAIL: is_vector test")

    def test_007_is_group(self):
        """Tests the is_group method"""
        eval_group = {}
        print("")
        print("is_group test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_group(test)
                print("PASS: " + name)
                eval_group[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_group[name] = 0
        self.assertTrue(eval_group == self.ans_group, "FAIL: is_group test")

    def test_008_is_dict_group(self):
        """Test the is_dict_group method"""
        eval_dict_group = {}
        print("")
        print("is_dict_group test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_dict_group(test)
                print("PASS: " + name)
                eval_dict_group[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_dict_group[name] = 0
        self.assertTrue(eval_dict_group == self.ans_dict_group, "FAIL: is_dict_group test")

    def test_009_is_series(self):
        """Test the is_series method"""
        eval_series = {}
        print("")
        print("is_series test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_series(test)
                print("PASS: " + name)
                eval_series[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_series[name] = 0
        # self.assertTrue(eval_dict_group == self.ans_series, "FAIL: is_dict_group test")
        self.assertDictEqual(eval_series, self.ans_series, "FAIL: is_series test")

    def test_010_is_numeric(self):
        """Tests the is_numeric method"""
        eval_numeric = {}
        print("")
        print("is_numeric test")
        print("-" * 70)
        for name, test in self.inputs.items():
            try:
                assert is_numeric(test)
                print("PASS: " + name)
                eval_numeric[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_numeric[name] = 0
        self.assertTrue(eval_numeric == self.ans_numeric, "FAIL: is_numeric test")

    def test_011_is_number(self):
        """Test the is_number function"""
        eval_numeric = {}
        for name, test in self.inputs.items():
            try:
                assert is_number(test)
                print("PASS: " + name)
                eval_numeric[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_numeric[name] = 0
        self.assertTrue(eval_numeric == self.ans_number, "FAIL: is_number test")

    def test_050_to_float_list(self):
        """Test the to_float int list conversion"""
        input_float = range(5)
        out_float = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(to_float(input_float), out_float, "FAIL: Error to_float int list")

    def test_051_to_float_quoted_list(self):
        """Test the to_float string quoted num list conversion"""
        input_float = ["1", "2", "3.0", "4.5", "5.65"]
        out_float = [1.0, 2.0, 3.0, 4.5, 5.65]
        self.assertEqual(to_float(input_float), out_float, "FAIL: Error to_float quoted string num list")

    def test_052_to_float_str_list(self):
        """Test the to_float string list conversion"""
        input_float = ["one", "two", "three", "four", "five"]
        out_float = [float("nan")] * 5
        self.assertTrue(np.array_equal(np.isnan(to_float(input_float)), np.isnan(out_float)),
                        "FAIL: Error to_float string list")

    def test_053_to_float_mixed_list(self):
        """Test the to_float mixed list conversion"""
        input_float = [1, "two", "3.0", 4.1, "5.65"]
        out_float = [1.0, float("nan"), 3.0, 4.1, 5.65]
        self.assertEqual([y for y in to_float(input_float) if not np.isnan(y)],
                         [x for x in out_float if not np.isnan(x)],
                         "FAIL: Error to_float mixed list")

    def test_054_to_float_missing_val_list(self):
        """Test the to_float missing val list conversion"""
        input_float = ["1.4", "", 3.0, 4, ""]
        out_float = [1.4, float("nan"), 3.0, 4, float("nan")]
        self.assertEqual([y for y in to_float(input_float) if not np.isnan(y)],
                         [x for x in out_float if not np.isnan(x)],
                         "FAIL: Error to_float missing val list")

    def test_055_to_float_empty_list(self):
        """Test the to_float empty list conversion"""
        input_float = []
        out_float = []
        self.assertEqual(to_float(input_float), out_float, "FAIL: Error to_float empty list")

        # Test flatten function

    def test_060_flatten_2_dim(self):
        """Test the flatten method on a 2 dim array"""
        input_flatten = [[1, 2, 3], [4, 5, 6]]
        out_flatten = [1, 2, 3, 4, 5, 6]
        self.assertTrue(np.array_equal(flatten(input_flatten), out_flatten), "FAIL: Error in flatten 2dim")

    def test_061_flatten_3_dim(self):
        """Test the flatten method on a 3 dim array"""
        input_flatten = [[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
        self.assertTrue(np.array_equal(flatten(input_flatten), out_flatten), "FAIL: Error in flatten 3dim")

    def test_062_flatten_4_dim(self):
        """Test the flatten method on a 4 dim array"""
        input_flatten = [[[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]],
                         [[[111, 112, 113], [114, 115, 116]], [[1111, 1112, 1113], [1114, 1115, 1116]]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16,
                       111, 112, 113, 114, 115, 116, 1111, 1112, 1113, 1114, 1115, 1116]
        self.assertTrue(np.array_equal(flatten(input_flatten), out_flatten), "FAIL: Error in flatten 4dim")


if __name__ == '__main__':
    unittest.main()
