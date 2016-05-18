"""sci_analysis test module
To run: python sci_analysis/test/test.py
"""
import unittest

import numpy.ma as ma
import numpy as np
import pandas as pd

from sci_analysis.operations.data_operations import is_array, is_dict, is_iterable, is_tuple, is_data, is_vector, \
    is_group, is_dict_group, drop_nan, to_float, flatten
#analysis import analyze
from sci_analysis.data.vector import Vector


class OperationsTestCasees(unittest.TestCase):
    """Tests all the methods in the data_operations module"""

    inputs = {
        'num': 3,
        'string': "hello",
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
        'group_of_lists': [range(5), range(6,10), range(11,15)],
        'dict_of_lists': {'a': range(1,5), 'b': range(6,10), 'c': range(11,15)}
    }

    ans_array = {
        'num': 0,
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

    def test_1_is_array(self):
        """Tests the is_array method"""
        eval_array = {}
        print("")
        print("is_array test")
        print("-" * 80)
        for name, test in self.inputs.iteritems():
            try:
                assert is_array(test)
                print("PASS: " + name)
                eval_array[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_array[name] = 0
        self.assertTrue(eval_array == self.ans_array, "FAIL: is_array test")

    def test_2_is_dict(self):
        """Tests the is_dict method"""
        eval_dict = {}
        print("")
        print("is_dict test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_dict(test)
                print("PASS: " + name)
                eval_dict[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_dict[name] = 0
        self.assertTrue(eval_dict == self.ans_dict, "FAIL: is_dict test")

    def test_3_is_iterable(self):
        """Tests the is_iterable method"""
        eval_iterable = {}
        print("")
        print("is_iterable test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_iterable(test)
                print("PASS: " + name)
                eval_iterable[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_iterable[name] = 0
        self.assertTrue(eval_iterable == self.ans_iterable, "FAIL: is_iterable test")

    def test_4_is_tuple(self):
        """Tests the is_tuple method"""
        eval_tuple = {}
        print("")
        print("is_tuple test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_tuple(test)
                print("PASS: " + name)
                eval_tuple[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_tuple[name] = 0
        self.assertTrue(eval_tuple == self.ans_tuple, "FAIL: is_tuple test")

    def test_5_is_data(self):
        """Tests the is_data method"""
        eval_data = {}
        print("")
        print("is_data test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_data(test)
                print("PASS: " + name)
                eval_data[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_data[name] = 0
        self.assertTrue(eval_data == self.ans_data, "FAIL: is_data test")

    def test_6_is_vector(self):
        """Tests the is_vector method"""
        eval_vector = {}
        print("")
        print("is_vector test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_vector(test)
                print("PASS: " + name)
                eval_vector[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_vector[name] = 0
        self.assertTrue(eval_vector == self.ans_vector, "FAIL: is_vector test")

    def test_7_is_group(self):
        """Tests the is_group method"""
        eval_group = {}
        print("")
        print("is_group test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_group(test)
                print("PASS: " + name)
                eval_group[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_group[name] = 0
        self.assertTrue(eval_group == self.ans_group, "FAIL: is_group test")

    def test_8_is_dict_group(self):
        """Test the is_dict_group method"""
        eval_dict_group = {}
        print("")
        print("is_dict_group test")
        print("-" * 70)
        for name, test in self.inputs.iteritems():
            try:
                assert is_dict_group(test)
                print("PASS: " + name)
                eval_dict_group[name] = 1
            except AssertionError:
                print("FAIL: " + name)
                eval_dict_group[name] = 0
        self.assertTrue(eval_dict_group == self.ans_dict_group, "FAIL: is_dict_group test")

    def test_9_to_float_list(self):
        """Test the to_float int list conversion"""
        input_float = range(5)
        out_float = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertTrue(to_float(input_float) == out_float, "FAIL: Error to_float int list")

    def test_10_to_float_quoted_list(self):
        """Test the to_float string quoted num list conversion"""
        input_float = ["1", "2", "3.0", "4.5", "5.65"]
        out_float = [1.0, 2.0, 3.0, 4.5, 5.65]
        self.assertTrue(to_float(input_float) == out_float, "FAIL: Error to_float quoted string num list")

    def test_11_to_float_str_list(self):
        """Test the to_float string list conversion"""
        input_float = ["one", "two", "three", "four", "five"]
        out_float = [float("nan")] * 5
        self.assertTrue(np.array_equal(np.isnan(to_float(input_float)), np.isnan(out_float)), "FAIL: Error to_float string list")

    def test_12_to_float_mixed_list(self):
        """Test the to_float mixed list conversion"""
        input_float = [1, "two", "3.0", 4.1, "5.65"]
        out_float = [1.0, float("nan"), 3.0, 4.1, 5.65]
        self.assertTrue([y for y in to_float(input_float) if not np.isnan(y)] == [x for x in out_float if not np.isnan(x)], "FAIL: Error to_float mixed list")

    def test_13_to_float_missing_val_list(self):
        """Test the to_float missing val list conversion"""
        input_float = ["1.4", "", 3.0, 4, ""]
        out_float = [1.4, float("nan"), 3.0, 4, float("nan")]
        self.assertTrue([y for y in to_float(input_float) if not np.isnan(y)] == [x for x in out_float if not np.isnan(x)], "FAIL: Error to_float missing val list")

    def test_14_to_float_empty_list(self):
        """Test the to_float empy list conversion"""
        input_float = []
        out_float = []
        self.assertTrue(to_float(input_float) == out_float, "FAIL: Error to_float empty list")

    def test_15_flatten_2_dim(self):
        """Test the flatten method on a 2 dim array"""
        input_flatten = [[1, 2, 3],[4, 5, 6]]
        out_flatten = [1, 2, 3, 4, 5, 6]
        self.assertTrue(flatten(input_flatten) == out_flatten, "FAIL: Error in flatten 2dim")

    def test_16_flatten_3_dim(self):
        """Test the flatten method on a 3 dim array"""
        input_flatten = [[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
        self.assertTrue(flatten(input_flatten) == out_flatten, "FAIL: Error in flatten 3dim")

    def test_17_flatten_4_dim(self):
        """Test the flatten method on a 4 dim array"""
        input_flatten = [[[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]], [[[111, 112, 113], [114, 115, 116]], [[1111, 1112, 1113], [1114, 1115, 1116]]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 111, 112, 113, 114, 115, 116, 1111, 1112, 1113, 1114, 1115, 1116]
        self.assertTrue(flatten(input_flatten) == out_flatten, "FAIL: Error in flatten 4dim")

    def test_18_create_vector_mixed_list(self):
        """Test vector creation from a mixed list"""
        input_array = [1.0, "2", '3.0', "four", 5.65]
        out_array = np.array([1.0, 2.0, 3.0, float("nan"), 5.65])
        self.assertTrue([y for y in Vector(input_array).data if not np.isnan(y)] == [x for x in out_array if not np.isnan(x)], "FAIL: Error in mixed list vector creation")

    def test_19_create_vector_missing_val(self):
        """Test vector creation from a missing value list"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, float("nan"), 3.0, 4.1, float("nan")])
        self.assertTrue([y for y in Vector(input_array).data if not np.isnan(y)] == [x for x in out_array if not np.isnan(x)], "FAIL: Error in missing val list vector creation")

    def test_20_create_vector_empty_list(self):
        """Test vector creation from an empty list"""
        self.assertTrue(not Vector().data, "FAIL: Error in empty list vector creation")

    def test_21_create_vector_2dim_array(self):
        """Test vector creation from a 2dim list"""
        input_array = np.array([[1, 2, 3], [1, 2, 3]])
        out_array = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(Vector(input_array).data, out_array), "FAIL: Error in 2dim array vector creation")

    def test_22_create_vector_dict(self):
        """Test vector creation from a dict"""
        input_array = {"one": 1, "two": 2.0, "three": "3", "four": "four"}
        out_array = [1., 2., 3., float("nan")]
        self.assertTrue(sorted([y for y in Vector(input_array).data if not np.isnan(y)]) == sorted([x for x in out_array if not np.isnan(x)]), "FAIL: Error in dict vector creation")

    def test_23_create_vector_tuple(self):
        """Test vector creation from a tuple"""
        input_array = (1, 2, 3, 4, 5)
        out_array = np.array([1., 2., 3., 4., 5.])
        self.assertTrue(np.array_equal(Vector(input_array).data, out_array), "FAIL: Error in tuple vector creation")

    def test_24_vector_is_empty(self):
        """Test the vector is_empty method"""
        input_array = []
        self.assertTrue(Vector(input_array).is_empty(), "FAIL: Error vector is_empty")

    def test_25_drop_nan(self):
        """Test the drop_nan method"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, 3.0, 4.1])
        self.assertTrue(np.array_equal(drop_nan(Vector(input_array)), out_array), "FAIL: Error in drop_nan")

    def test_26_drop_nan_intersect(self):
        """Test the drop_nan_intersect method"""
        input_array_1 = [1., float("nan"), 3., float("nan"), 5.]
        input_array_2 = [11., float("nan"), 13., 14., 15.]
        output_array = ()



"""



# Test individual tests
#norm = NormTest(inputs['large_array'], display=False)


# Test Group Analysis
a = np.random.rand(50) * 2
b = np.random.rand(50) * 3
c = np.random.rand(50)
d = np.random.rand(50) * 4
e = np.random.rand(50) * 2


# Test analysis function
try:
    assert analyze(xdata=d, name="Test")
    print "Pass histo test"
except AssertionError:
    print "Fail histo test"

try:
    assert analyze([a, b, c, d, e], groups=["A", "B", "C", "D", "E"])
    print "Pass group list test"
except AssertionError:
    print "Fail group list test"

try:
    assert analyze({"A": a, "B": b, "C": c, "D": d, "E": e})
    print "Pass group dict test"
except AssertionError:
    print "Fail group dict test"

#analyze([a, b, c, d, e], groups=["A", "B", "C", "D", "E"])
#analyze({"A": a, "B": b, "C": c, "D": d, "E": e})
#analyze(xdata=d, name="Test")
"""

if __name__ == '__main__':
    unittest.main()