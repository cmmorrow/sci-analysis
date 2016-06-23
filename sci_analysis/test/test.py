"""sci_analysis test module
To run: python sci_analysis/test/test.py
"""
import unittest

import numpy.ma as ma
import numpy as np
import pandas as pd
import scipy.stats as st

from sci_analysis.operations.data_operations import is_array, is_dict, is_iterable, is_tuple, is_data, is_vector, \
    is_group, is_dict_group, drop_nan, drop_nan_intersect, to_float, flatten
#analysis import analyze
from sci_analysis.data.data import Data
from sci_analysis.data.vector import Vector
from sci_analysis.analysis import TTest, KSTest, NormTest, LinearRegression, Correlation, EqualVariance, Kruskal, \
    Anova, VectorStatistics


class SciAnalysisTest(unittest.TestCase):
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
        self.assertTrue(eval_array == self.ans_array, "FAIL: is_array test")

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

# Test to_float function

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
        self.assertEqual(flatten(input_flatten), out_flatten, "FAIL: Error in flatten 2dim")

    def test_061_flatten_3_dim(self):
        """Test the flatten method on a 3 dim array"""
        input_flatten = [[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
        self.assertEqual(flatten(input_flatten), out_flatten, "FAIL: Error in flatten 3dim")

    def test_062_flatten_4_dim(self):
        """Test the flatten method on a 4 dim array"""
        input_flatten = [[[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]],
                         [[[111, 112, 113], [114, 115, 116]], [[1111, 1112, 1113], [1114, 1115, 1116]]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16,
                       111, 112, 113, 114, 115, 116, 1111, 1112, 1113, 1114, 1115, 1116]
        self.assertEqual(flatten(input_flatten), out_flatten, "FAIL: Error in flatten 4dim")

# Test vector creation

    def test_100_create_vector_mixed_list(self):
        """Test vector creation from a mixed list"""
        input_array = [1.0, "2", '3.0', "four", 5.65]
        out_array = np.array([1.0, 2.0, 3.0, float("nan"), 5.65])
        self.assertEqual([y for y in Vector(input_array).data if not np.isnan(y)],
                         [x for x in out_array if not np.isnan(x)],
                         "FAIL: Error in mixed list vector creation")

    def test_101_create_vector_missing_val(self):
        """Test vector creation from a missing value list"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, float("nan"), 3.0, 4.1, float("nan")])
        self.assertEqual([y for y in Vector(input_array).data if not np.isnan(y)],
                         [x for x in out_array if not np.isnan(x)],
                         "FAIL: Error in missing val list vector creation")

    def test_102_create_vector_empty_list(self):
        """Test vector creation from an empty list"""
        self.assertTrue(not Vector().data, "FAIL: Error in empty list vector creation")

    def test_103_create_vector_2dim_array(self):
        """Test vector creation from a 2dim list"""
        input_array = np.array([[1, 2, 3], [1, 2, 3]])
        out_array = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(Vector(input_array).data, out_array),
                        "FAIL: Error in 2dim array vector creation")

    def test_104_create_vector_dict(self):
        """Test vector creation from a dict"""
        input_array = {"one": 1, "two": 2.0, "three": "3", "four": "four"}
        out_array = [1., 2., 3., float("nan")]
        self.assertEqual(sorted([y for y in Vector(input_array).data if not np.isnan(y)]),
                         sorted([x for x in out_array if not np.isnan(x)]),
                         "FAIL: Error in dict vector creation")

    def test_105_create_vector_tuple(self):
        """Test vector creation from a tuple"""
        input_array = (1, 2, 3, 4, 5)
        out_array = np.array([1., 2., 3., 4., 5.])
        self.assertTrue(np.array_equal(Vector(input_array).data, out_array), "FAIL: Error in tuple vector creation")

    def test_106_create_vector_0d(self):
        """Test vector creation from a 0 dimension array"""
        input_array = np.array(4)
        self.assertTrue(float(Vector(input_array).data) == 4), "FAIL: Error vector creation from 0d array"

    def test_120_create_vector_none(self):
        """Test vector creation from None"""
        self.assertTrue(Vector(None).is_empty(), "FAIL: Error vector creation from None")

# Test vector is_empty method

    def test_121_vector_is_empty_empty_list(self):
        """Test the vector is_empty method"""
        input_array = []
        self.assertTrue(Vector(input_array).is_empty(), "FAIL: Error vector is_empty")

    def test_122_vector_is_empty_empty_array(self):
        """Test the vector is_empty method"""
        input_array = np.array([])
        self.assertTrue(Vector(input_array).is_empty(), "FAIL: Error vector is_empty")

# Test drop nan functions

    def test_122_drop_nan(self):
        """Test the drop_nan method"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, 3.0, 4.1])
        self.assertTrue(np.array_equal(drop_nan(Vector(input_array)), out_array), "FAIL: Error in drop_nan")

    def test_123_drop_nan_intersect(self):
        """Test the drop_nan_intersect method"""
        input_array_1 = [1., float("nan"), 3., float("nan"), 5.]
        input_array_2 = [11., float("nan"), 13., 14., 15.]
        output_array = [(1., 11.), (3., 13.), (5., 15.)]
        inter = drop_nan_intersect(Vector(input_array_1), Vector(input_array_2))
        self.assertTrue(zip(inter[0], inter[1]) == output_array, "FAIL: Error in drop_nan_intersect")

    def test_124_drop_nan_empty(self):
        """Test the drop_nan method on an empty array"""
        input_array = np.array([])
        self.assertFalse(drop_nan(input_array), "FAIL: Error in drop_nan empty array")

# Test TTest

    def test_200_TTest_single_matched(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        alpha = 0.05
        self.assertTrue(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).p_value > alpha,
                        "FAIL: TTest single type I error")

    def test_201_TTest_single_matched_test_type(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).test_type, '1_sample',
                         "FAIL: TTest incorrect test type")

    def test_202_TTest_single_matched_mu(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).mu, y_val,
                         "FAIL: TTest incorrect mu")

    def test_203_TTest_single_matched_t_value(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertTrue(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).t_value,
                        "FAIL: TTest t value not set")

    def test_204_TTest_single_matched_statistic(self):
        """Test the TTest against a given matched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 4.0
        self.assertTrue(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).statistic,
                        "FAIL: TTest statistic not set")

    def test_205_TTest_single_unmatched(self):
        """Test the TTest against a given unmatched value"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_val = 5.0
        alpha = 0.05
        self.assertFalse(TTest(st.norm.rvs(*x_parms, size=100), y_val, display=False).p_value > alpha,
                         "FAIL: TTest single type II error")

    def test_206_TTest_equal_variance_matched(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        alpha = 0.05
        self.assertGreater(TTest(st.norm.rvs(*x_parms, size=100),
                                 st.norm.rvs(*y_parms, size=100), display=False).p_value,
                           alpha, "FAIL: TTest equal variance matched Type I error")

    def test_207_TTest_equal_variance_matched_test_type(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100),
                               st.norm.rvs(*y_parms, size=100), display=False).test_type, 't_test',
                         "FAIL: TTest incorrect test type")

    def test_208_TTest_equal_variance_matched_t_value(self):
        """Test the TTest with two samples with equal variance and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        self.assertAlmostEqual(TTest(st.norm.rvs(*x_parms, size=100),
                                     st.norm.rvs(*y_parms, size=100), display=False).t_value, -0.2592,
                               msg="FAIL: TTest equal variance matched Type I error", delta=0.0001)

    def test_209_TTest_equal_variance_unmatched(self):
        """Test the TTest with two samples with equal variance and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 0.75]
        alpha = 0.05
        self.assertLess(TTest(st.norm.rvs(*x_parms, size=100),
                              st.norm.rvs(*y_parms, size=100), display=False).p_value, alpha,
                        "FAIL: TTest equal variance unmatched Type II error")

    def test_210_TTest_unequal_variance_matched(self):
        """Test the TTest with two samples with different variances and matched means"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 1.35]
        alpha = 0.05
        self.assertGreater(TTest(st.norm.rvs(*x_parms, size=100),
                                 st.norm.rvs(*y_parms, size=100), display=False).p_value, alpha,
                           "FAIL: TTest different variance matched Type I error")

    def test_211_TTest_unequal_variance_unmatched(self):
        """Test the TTest with two samples with different variances and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 1.12]
        alpha = 0.05
        self.assertLess(TTest(st.norm.rvs(*x_parms, size=100),
                              st.norm.rvs(*y_parms, size=100), display=False).p_value, alpha,
                        "FAIL: TTest different variance unmatched Type II error")

    def test_212_TTest_unequal_variance_unmatched_test_type(self):
        """Test the TTest with two samples with different variances and different means"""
        np.random.seed(987654321)
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 1.12]
        self.assertEqual(TTest(st.norm.rvs(*x_parms, size=100),
                               st.norm.rvs(*y_parms, size=100), display=False).test_type, 'welch_t',
                         "FAIL: TTest incorrect test type")

# Test KSTest

    def test_250_Kolmogorov_Smirnov_normal_test(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertGreater(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Error in norm GOF")

    def test_251_Kolmogorov_Smirnov_normal_test_distribution_type(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertEqual(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).distribution, distro,
                         "FAIL: KSTest distribution not set")

    def test_252_Kolmogorov_Smirnov_normal_test_statistic(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertTrue(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).statistic,
                        "FAIL: KSTest statistic not set")

    def test_253_Kolmogorov_Smirnov_normal_test_D_value(self):
        """Test the normal distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'norm'
        self.assertTrue(KSTest(st.norm.rvs(size=100), distro, alpha=alpha, display=False).d_value,
                        "FAIL: KSTest d_value not set")

    def test_254_Kolmogorov_Smirnov_alpha_test_parms_missing(self):
        """Test the KSTest to make sure an exception is raised if parms are missing"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'alpha'  # not to be confused with the sensitivity alpha
        self.assertRaises(TypeError, lambda: KSTest(st.alpha.rvs(size=100), distro, alpha=alpha, display=False),
                          "FAIL: missing parms does not raise exception")

    def test_255_Kolmogorov_Smirnov_alpha_test(self):
        """Test the alpha distribution detection"""
        np.random.seed(987654321)
        parms = [3.5]
        alpha = 0.05
        distro = 'alpha'
        self.assertGreater(KSTest(st.alpha.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha, "FAIL: Error in alpha GOF")

    def test_256_Kolmogorov_Smirnov_beta_test(self):
        """Test the beta distribution detection"""
        np.random.seed(987654321)
        parms = [2.3, 0.6]
        alpha = 0.05
        distro = 'beta'
        self.assertGreater(KSTest(st.beta.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha, "FAIL: Error in beta GOF")

    def test_257_Kolmogorov_Smirnov_cauchy_test(self):
        """Test the cauchy distribution detection"""
        np.random.seed(987654321)
        alpha = 0.05
        distro = 'cauchy'
        self.assertGreater(KSTest(st.cauchy.rvs(size=100), distro,
                                  alpha=alpha, display=False).p_value, alpha, "FAIL: Error in cauchy GOF")

    def test_258_Kolmogorov_Smirnov_chi2_large_test(self):
        """Test the chi squared distribution detection with sufficiently large dof"""
        np.random.seed(987654321)
        parms = [50]
        alpha = 0.05
        distro = 'chi2'
        self.assertGreater(KSTest(st.chi2.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Error in chi2 large GOF")

    def test_259_Kolmogorov_Smirnov_chi2_small_test(self):
        """Test the chi squared distribution detection with small dof"""
        np.random.seed(987654321)
        parms = [5]
        alpha = 0.05
        distro = 'chi2'
        self.assertGreater(KSTest(st.chi2.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Error in chi2 small GOF")

    def test_260_Kolmogorov_Smirnov_weibull_min_test(self):
        """Test the weibull min distribution detection"""
        np.random.seed(987654321)
        parms = [1.7]
        alpha = 0.05
        distro = 'weibull_min'
        self.assertGreater(KSTest(st.weibull_min.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Error in Weibull min GOF")

    def test_261_Kolmogorov_Smirnov_weibull_max_test(self):
        """Test the weibull min distribution detection"""
        np.random.seed(987654321)
        parms = [2.8]
        alpha = 0.05
        distro = 'weibull_max'
        self.assertGreater(KSTest(st.weibull_max.rvs(*parms, size=100), distro,
                                  parms=parms, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Error in Weibull max GOF")

# Test NormTest

    def test_300_Norm_test_single(self):
        """Test the normal distribution check"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertGreater(NormTest(st.norm.rvs(*parms, size=100), display=False, alpha=alpha).p_value, alpha,
                           "FAIL: Normal test Type I error")

    def test_301_Norm_test_single_fail(self):
        """Test the normal distribution check fails for a different distribution"""
        np.random.seed(987654321)
        parms = [1.7]
        alpha = 0.05
        self.assertLess(NormTest(st.weibull_min.rvs(*parms, size=100), alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Normal test Type II error")

    def test_302_Norm_test_statistic(self):
        """Test the normal distribution statistic value is set"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertTrue(NormTest(st.norm.rvs(*parms, size=100), alpha=alpha, display=False).statistic,
                        "FAIL: Normal test statistic not set")

    def test_303_Norm_test_W_value(self):
        """Test the normal distribution W value is set"""
        np.random.seed(987654321)
        parms = [5, 0.1]
        alpha = 0.05
        self.assertTrue(NormTest(st.norm.rvs(*parms, size=100), alpha=alpha, display=False).w_value,
                        "FAIL: Normal test W value not set")

    def test_304_Norm_test_multi_pass(self):
        """Test if multiple vectors are from the normal distribution"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), st.norm.rvs(4, 0.75, size=75), st.norm.rvs(1, 1, size=50)]
        self.assertGreater(NormTest(*groups, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Normal test Type I error")

    def test_305_Norm_test_multi_fail(self):
        """Test if multiple vectors are from the normal distribution, with one failing"""
        np.random.seed(987654321)
        alpha = 0.05
        groups = [st.norm.rvs(5, 0.1, size=100), st.weibull_min.rvs(1.7, size=75), st.norm.rvs(1, 1, size=50)]
        self.assertLess(NormTest(*groups, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Normal test Type II error")

# Test LinearRegression

    def test_350_LinRegress_corr(self):
        """Test the Linear Regression class for correlation"""
        np.random.seed(987654321)
        x_input_array = range(1, 101)
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertLess(LinearRegression(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Linear Regression Type II error")

    def test_351_LinRegress_no_corr(self):
        """Test the Linear Regression class for uncorrelated data"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertGreater(LinearRegression(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Linear Regression Type I error")

    def test_352_LinRegress_no_corr_slope(self):
        """Test the Linear Regression slope"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).slope, -0.0969, delta=0.0001,
                               msg="FAIL: Linear Regression slope")

    def test_353_LinRegress_no_corr_intercept(self):
        """Test the Linear Regression intercept"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).intercept, -0.0397, delta=0.0001,
                               msg="FAIL: Linear Regression intercept")

    def test_354_LinRegress_no_corr_r2(self):
        """Test the Linear Regression R^2"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).r_squared, -0.1029, delta=0.0001,
                               msg="FAIL: Linear Regression R^2")

    def test_355_LinRegress_no_corr_std_err(self):
        """Test the Linear Regression std err"""
        np.random.seed(987654321)
        alpha = 0.05
        x_input_array = st.norm.rvs(size=200)
        y_input_array = st.norm.rvs(size=200)
        self.assertAlmostEqual(LinearRegression(x_input_array, y_input_array,
                                                alpha=alpha,
                                                display=False).std_err, 0.0666, delta=0.0001,
                               msg="FAIL: Linear Regression std err")

# Test Correlation

    def test_400_Correlation_corr_pearson(self):
        """Test the Correlation class for correlated normally distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertLess(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Correlation pearson Type II error")

    def test_401_Correlation_corr_pearson_test_type(self):
        """Test the Correlation class for correlated normally distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.norm.rvs(size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertEqual(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).test_type, 'pearson',
                         "FAIL: Correlation pearson wrong type")

    def test_402_Correlation_no_corr_pearson(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertGreater(Correlation(st.norm.rvs(size=100),
                                       st.norm.rvs(size=100),
                                       alpha=alpha,
                                       display=False).p_value, alpha,
                           "FAIL: Correlation pearson Type I error")

    def test_403_Correlation_no_corr_pearson_test_type(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertEqual(Correlation(st.norm.rvs(size=100),
                                     st.norm.rvs(size=100),
                                     alpha=alpha,
                                     display=False).test_type, 'pearson',
                         "FAIL: Correlation pearson wrong type")

    def test_404_Correlation_no_corr_pearson_r_value(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        np.random.seed(987654321)
        alpha = 0.05
        self.assertAlmostEqual(Correlation(st.norm.rvs(size=100),
                               st.norm.rvs(size=100),
                               alpha=alpha,
                               display=False).r_value, -0.0055,
                               delta=0.0001,
                               msg="FAIL: Correlation pearson r value")

    def test_405_Correlation_corr_spearman(self):
        """Test the Correlation class for correlated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.weibull_min.rvs(1.7, size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertLess(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: Correlation spearman Type II error")

    def test_406_Correlation_corr_spearman_test_type(self):
        """Test the Correlation class for correlated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = list(st.weibull_min.rvs(1.7, size=100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertEqual(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).test_type, 'spearman',
                         "FAIL: Correlation spearman wrong type")

    def test_407_Correlation_no_corr_spearman(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertGreater(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).p_value, alpha,
                           "FAIL: Correlation spearman Type I error")

    def test_408_Correlation_no_corr_spearman_test_type(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertEqual(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).test_type, 'spearman',
                         "FAIL: Correlation spearman wrong type")

    def test_409_Correlation_no_corr_spearman_xdata(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).xdata,
                                       x_input_array),
                        "FAIL: Correlation spearman xdata")

    def test_410_Correlation_no_corr_spearman_predictor(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).predictor,
                                       x_input_array),
                        "FAIL: Correlation spearman predictor")

    def test_411_Correlation_no_corr_spearman_ydata(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).ydata,
                                       y_input_array),
                        "FAIL: Correlation spearman ydata")

    def test_412_Correlation_no_corr_spearman_response(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        np.random.seed(987654321)
        x_input_array = st.norm.rvs(size=100)
        y_input_array = st.weibull_min.rvs(1.7, size=100)
        alpha = 0.05
        self.assertTrue(np.array_equal(Correlation(x_input_array, y_input_array, alpha=alpha, display=False).response,
                                       y_input_array),
                        "FAIL: Correlation spearman response")

# Test EqualVariance

    def test_450_EqualVariance_Bartlett_matched(self):
        """Test the EqualVariance class for normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertGreater(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value,
                           a,
                           "FAIL: Equal variance Bartlett Type I error")

    def test_451_EqualVariance_Bartlett_matched_test_type(self):
        """Test the EqualVariance class for normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Bartlett",
                         "FAIL: Equal variance Bartlett test type")

    def test_452_EqualVariance_Bartlett_unmatched(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertLess(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value, a,
                        "FAIL: Equal variance bartlett Type II error")

    def test_453_EqualVariance_Bartlett_unmatched_test_type(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Bartlett",
                         "FAIL: Equal variance bartlett test type")

    def test_454_EqualVariance_Bartlett_unmatched_statistic(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(EqualVariance(x_input_array, y_input_array, z_input_array,
                                             alpha=a,
                                             display=False).statistic,
                               43.0402,
                               delta=0.0001,
                               msg="FAIL: Equal variance bartlett statistic")

    def test_455_EqualVariance_Bartlett_unmatched_t_value(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(EqualVariance(x_input_array, y_input_array, z_input_array,
                                             alpha=a,
                                             display=False).t_value,
                               43.0402,
                               delta=0.0001,
                               msg="FAIL: Equal variance bartlett t value")

    def test_456_EqualVariance_Bartlett_unmatched_w_value(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        a = 0.05
        self.assertRaises(KeyError, lambda: EqualVariance(x_input_array, y_input_array, z_input_array,
                                                          alpha=a,
                                                          display=False).w_value)

    # TODO: Update this to use a specific exception in the future
    def test_457_EqualVariance_Bartlett_single_argument(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [4, 1.35]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(TypeError, lambda: EqualVariance(x_input_array, alpha=a, display=False).p_value)

    def test_458_EqualVariance_Levene_matched(self):
        """Test the EqualVariance class for non-normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        z_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        a = 0.05
        self.assertGreater(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value,
                           a,
                           "FAIL: Unequal variance levene Type I error")

    def test_459_EqualVariance_Levene_matched_test_type(self):
        """Test the EqualVariance class for non-normally distributed matched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [1.7]
        z_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        a = 0.05
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Levene",
                         "FAIL: Unequal variance levene test type")

    def test_460_EqualVariance_Levene_unmatched(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertLess(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).p_value, a,
                        "FAIL: Unequal variance levene Type II error")

    def test_461_EqualVariance_Levene_unmatched_test_type(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertEqual(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=a, display=False).test_type,
                         "Levene",
                         "FAIL: Unequal variance levene test type")

    def test_462_EqualVariance_Levene_unmatched_statistic(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertAlmostEqual(EqualVariance(x_input_array,
                                             y_input_array,
                                             z_input_array,
                                             alpha=a,
                                             display=False).statistic,
                               11.2166,
                               delta=0.0001,
                               msg="FAIL: Unequal variance levene statistic")

    def test_463_EqualVariance_Levene_unmatched_w_value(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertAlmostEqual(EqualVariance(x_input_array,
                                             y_input_array,
                                             z_input_array,
                                             alpha=a,
                                             display=False).w_value,
                               11.2166,
                               delta=0.0001,
                               msg="FAIL: Unequal variance levene w value")

    def test_464_EqualVariance_Levene_unmatched_t_value(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        np.random.seed(987654321)
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        a = 0.05
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*y_parms, size=100)
        z_input_array = st.weibull_min.rvs(*z_parms, size=100)
        self.assertRaises(KeyError, lambda: EqualVariance(x_input_array,
                                                          y_input_array,
                                                          z_input_array,
                                                          alpha=a,
                                                          display=False).t_value)

# Test Kruskal

    def test_500_Kruskal_matched(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertGreater(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: Kruskal Type I error")

    def test_501_Kruskal_matched_statistic(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(Kruskal(x_input_array, y_input_array, z_input_array, alpha=a, display=False).statistic,
                               0.4042,
                               delta=0.0001,
                               msg="FAIL: Kruskal statistic")

    def test_502_Kruskal_matched_h_value(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        y_input_array = st.weibull_min.rvs(*x_parms, size=100)
        z_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertAlmostEqual(Kruskal(x_input_array, y_input_array, z_input_array, alpha=a, display=False).h_value,
                               0.4042,
                               delta=0.0001,
                               msg="FAIL: Kruskal h value")

    def test_503_Kruskal_matched_single_argument(self):
        """Test the Kruskal Wallis class on matched data"""
        np.random.seed(987654321)
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=100)
        a = 0.05
        self.assertRaises(TypeError, lambda: Kruskal(x_input_array, alpha=a, display=False).p_value)

    def test_504_Kruskal_unmatched(self):
        """Test the Kruskal Wallis class on unmatched data"""
        np.random.seed(987654321)
        x_parms = [1.7, 1]
        z_parms = [0.8, 1]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*z_parms, size=100)
        alpha = 0.05
        self.assertLess(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                        alpha,
                        "FAIL: Kruskal Type II error")

# Test ANOVA

    def test_550_ANOVA_matched(self):
        """Test the ANOVA class on matched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertGreater(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value,
                           alpha,
                           "FAIL: ANOVA Type I error")

    def test_551_ANOVA_matched_statistic(self):
        """Test the ANOVA class on matched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertAlmostEqual(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).statistic,
                               0.1076,
                               delta=0.0001,
                               msg="FAIL: ANOVA statistic")

    def test_552_ANOVA_matched_f_value(self):
        """Test the ANOVA class on matched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=100)
        y_input_array = st.norm.rvs(*x_parms, size=100)
        z_input_array = st.norm.rvs(*x_parms, size=100)
        alpha = 0.05
        self.assertAlmostEqual(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).f_value,
                               0.1076,
                               delta=0.0001,
                               msg="FAIL: ANOVA f value")

    def test_553_ANOVA_unmatched(self):
        """Test the ANOVA class on unmatched data"""
        np.random.seed(987654321)
        x_parms = [4, 1.75]
        y_parms = [6, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array =st.norm.rvs(*y_parms, size=1000)
        z_input_array = st.norm.rvs(*x_parms, size=1000)
        alpha = 0.05
        self.assertLess(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha, display=False).p_value, alpha,
                        "FAIL: ANOVA Type II error")

# Test VectorStatistics

    def test_1000_Vector_stats_count(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertEqual(VectorStatistics(input_array, sample=True, display=False).count, 100, "FAIL: Stat count")

    def test_1001_Vector_stats_mean(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).mean,
                               4.0145,
                               delta=0.0001,
                               msg="FAIL: Stat mean")

    def test_1002_Vector_stats_std_dev_sample(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).std_dev,
                               1.8622,
                               delta=0.0001,
                               msg="FAIL: Stat std dev")

    def test_1003_Vector_stats_std_dev_population(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=False, display=False).std_dev,
                               1.8529,
                               delta=0.0001,
                               msg="FAIL: Stat std dev")

    def test_1004_Vector_stats_std_error_sample(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).std_err,
                               0.1862,
                               delta=0.0001,
                               msg="FAIL: Stat std error")

    def test_1004_Vector_stats_std_error_population(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=False, display=False).std_err,
                               0.1853,
                               delta=0.0001,
                               msg="FAIL: Stat std error")

    def test_1005_Vector_stats_skewness(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).skewness,
                               -0.0256,
                               delta=0.0001,
                               msg="FAIL: Stat skewness")

    def test_1006_Vector_stats_kurtosis(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).kurtosis,
                               -0.4830,
                               delta=0.0001,
                               msg="FAIL: Stat kurtosis")

    def test_1007_Vector_stats_maximum(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).maximum,
                               7.9315,
                               delta=0.0001,
                               msg="FAIL: Stat maximum")

    def test_1008_Vector_stats_q3(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).q3,
                               5.0664,
                               delta=0.0001,
                               msg="FAIL: Stat q3")

    def test_1009_Vector_stats_median(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).median,
                               4.1331,
                               delta=0.0001,
                               msg="FAIL: Stat median")

    def test_1010_Vector_stats_q1(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).q1,
                               2.6576,
                               delta=0.0001,
                               msg="FAIL: Stat q1")

    def test_1011_Vector_stats_minimum(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).minimum,
                               -0.3256,
                               delta=0.0001,
                               msg="FAIL: Stat minimum")

    def test_1012_Vector_stats_range(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).range,
                               8.2571,
                               delta=0.0001,
                               msg="FAIL: Stat range")

    def test_1013_Vector_stats_iqr(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertAlmostEqual(VectorStatistics(input_array, sample=True, display=False).iqr,
                               2.4088,
                               delta=0.0001,
                               msg="FAIL: Stat iqr")

    def test_1014_Vector_stats_name(self):
        """Test the vector statistics class"""
        np.random.seed(987654321)
        parms = [4, 1.75]
        input_array = st.norm.rvs(*parms, size=100)
        self.assertEqual(VectorStatistics(input_array, sample=True, display=False).name,
                         "Statistics",
                         "FAIL: Stat name")

    def test_1015_Vector_stats_min_size(self):
        """Test the vector statistics class"""
        input_array = np.array([14])
        self.assertFalse(VectorStatistics(input_array, sample=True, display=False).data, "FAIL: Stats not None")

    def test_1016_Vector_stats_empty_array(self):
        """Test the vector statistics class"""
        self.assertFalse(VectorStatistics(np.array([]), sample=True, display=False).data,
                         "FAIL: Stats not None")


if __name__ == '__main__':
    unittest.main()
