"""sci_analysis test module
To run: python sci_analysis/test/test.py
"""
import unittest

import numpy.ma as ma
import numpy as np
import pandas as pd
import scipy.stats as st

from sci_analysis.operations.data_operations import is_array, is_dict, is_iterable, is_tuple, is_data, is_vector, \
    is_group, is_dict_group, drop_nan, to_float, flatten
#analysis import analyze
from sci_analysis.data.data import Data
from sci_analysis.data.vector import Vector
from sci_analysis.analysis import TTest, KSTest, NormTest, LinearRegression, Correlation, EqualVariance, Kruskal, \
    Anova, GroupNormTest, VectorStatistics


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

    def test_020_to_float_list(self):
        """Test the to_float int list conversion"""
        input_float = range(5)
        out_float = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertTrue(to_float(input_float) == out_float, "FAIL: Error to_float int list")

    def test_021_to_float_quoted_list(self):
        """Test the to_float string quoted num list conversion"""
        input_float = ["1", "2", "3.0", "4.5", "5.65"]
        out_float = [1.0, 2.0, 3.0, 4.5, 5.65]
        self.assertTrue(to_float(input_float) == out_float, "FAIL: Error to_float quoted string num list")

    def test_022_to_float_str_list(self):
        """Test the to_float string list conversion"""
        input_float = ["one", "two", "three", "four", "five"]
        out_float = [float("nan")] * 5
        self.assertTrue(np.array_equal(np.isnan(to_float(input_float)), np.isnan(out_float)), "FAIL: Error to_float string list")

    def test_023_to_float_mixed_list(self):
        """Test the to_float mixed list conversion"""
        input_float = [1, "two", "3.0", 4.1, "5.65"]
        out_float = [1.0, float("nan"), 3.0, 4.1, 5.65]
        self.assertTrue([y for y in to_float(input_float) if not np.isnan(y)] == [x for x in out_float if not np.isnan(x)], "FAIL: Error to_float mixed list")

    def test_024_to_float_missing_val_list(self):
        """Test the to_float missing val list conversion"""
        input_float = ["1.4", "", 3.0, 4, ""]
        out_float = [1.4, float("nan"), 3.0, 4, float("nan")]
        self.assertTrue([y for y in to_float(input_float) if not np.isnan(y)] == [x for x in out_float if not np.isnan(x)], "FAIL: Error to_float missing val list")

    def test_025_to_float_empty_list(self):
        """Test the to_float empy list conversion"""
        input_float = []
        out_float = []
        self.assertTrue(to_float(input_float) == out_float, "FAIL: Error to_float empty list")

    def test_040_flatten_2_dim(self):
        """Test the flatten method on a 2 dim array"""
        input_flatten = [[1, 2, 3],[4, 5, 6]]
        out_flatten = [1, 2, 3, 4, 5, 6]
        self.assertTrue(flatten(input_flatten) == out_flatten, "FAIL: Error in flatten 2dim")

    def test_041_flatten_3_dim(self):
        """Test the flatten method on a 3 dim array"""
        input_flatten = [[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
        self.assertTrue(flatten(input_flatten) == out_flatten, "FAIL: Error in flatten 3dim")

    def test_042_flatten_4_dim(self):
        """Test the flatten method on a 4 dim array"""
        input_flatten = [[[[1, 2, 3], [4, 5, 6]], [[11, 12, 13], [14, 15, 16]]], [[[111, 112, 113], [114, 115, 116]], [[1111, 1112, 1113], [1114, 1115, 1116]]]]
        out_flatten = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 111, 112, 113, 114, 115, 116, 1111, 1112, 1113, 1114, 1115, 1116]
        self.assertTrue(flatten(input_flatten) == out_flatten, "FAIL: Error in flatten 4dim")

    def test_060_create_vector_mixed_list(self):
        """Test vector creation from a mixed list"""
        input_array = [1.0, "2", '3.0', "four", 5.65]
        out_array = np.array([1.0, 2.0, 3.0, float("nan"), 5.65])
        self.assertTrue([y for y in Vector(input_array).data if not np.isnan(y)] == [x for x in out_array if not np.isnan(x)], "FAIL: Error in mixed list vector creation")

    def test_061_create_vector_missing_val(self):
        """Test vector creation from a missing value list"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, float("nan"), 3.0, 4.1, float("nan")])
        self.assertTrue([y for y in Vector(input_array).data if not np.isnan(y)] == [x for x in out_array if not np.isnan(x)], "FAIL: Error in missing val list vector creation")

    def test_062_create_vector_empty_list(self):
        """Test vector creation from an empty list"""
        self.assertTrue(not Vector().data, "FAIL: Error in empty list vector creation")

    def test_063_create_vector_2dim_array(self):
        """Test vector creation from a 2dim list"""
        input_array = np.array([[1, 2, 3], [1, 2, 3]])
        out_array = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        self.assertTrue(np.array_equal(Vector(input_array).data, out_array), "FAIL: Error in 2dim array vector creation")

    def test_064_create_vector_dict(self):
        """Test vector creation from a dict"""
        input_array = {"one": 1, "two": 2.0, "three": "3", "four": "four"}
        out_array = [1., 2., 3., float("nan")]
        self.assertTrue(sorted([y for y in Vector(input_array).data if not np.isnan(y)]) == sorted([x for x in out_array if not np.isnan(x)]), "FAIL: Error in dict vector creation")

    def test_065_create_vector_tuple(self):
        """Test vector creation from a tuple"""
        input_array = (1, 2, 3, 4, 5)
        out_array = np.array([1., 2., 3., 4., 5.])
        self.assertTrue(np.array_equal(Vector(input_array).data, out_array), "FAIL: Error in tuple vector creation")

    def test_066_create_vector_none(self):
        """Test vector creation from None"""
        input_array = None
        self.assertTrue(Vector(None).is_empty(), "FAIL: Error vector creation from None")

    def test_080_vector_is_empty(self):
        """Test the vector is_empty method"""
        input_array = []
        self.assertTrue(Vector(input_array).is_empty(), "FAIL: Error vector is_empty")

    def test_100_drop_nan(self):
        """Test the drop_nan method"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, 3.0, 4.1])
        self.assertTrue(np.array_equal(drop_nan(Vector(input_array)), out_array), "FAIL: Error in drop_nan")

    def test_101_drop_nan_intersect(self):
        """Test the drop_nan_intersect method"""
        input_array_1 = [1., float("nan"), 3., float("nan"), 5.]
        input_array_2 = [11., float("nan"), 13., 14., 15.]
        output_array = ()

    def test_102_TTest_single_matched(self):
        """Test the TTest against a given matched value"""
        x_parms = [4, 0.75]
        y_val = 4.0
        alpha = 0.05
        results = [True for _ in range(4) if TTest(st.norm.rvs(*x_parms, size=1000), y_val,
                                                   display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: TTest single type I error")

    def test_103_TTest_single_unmatched(self):
        """Test the TTest against a given unmatched value"""
        x_parms = [4, 0.75]
        y_val = 5.0
        alpha = 0.05
        results = [True for _ in range(4) if TTest(st.norm.rvs(*x_parms, size=1000), y_val,
                                                   display=False).results[0] > alpha]
        self.assertFalse(True if True in results else False, "FAIL: TTest single type II error")

    def test_120_Kolmogorov_Smirnov_normal_test(self):
        """Test the normal distribution detection"""
        alpha = 0.05
        distro = 'norm'
        results = [True for _ in range(4) if KSTest(st.norm.rvs(size=1000), distro, alpha=alpha,
                                                    display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in norm GOF")

    def test_121_Kolmogorov_Smirnov_alpha_test(self):
        """Test the alpha distribution detection"""
        parms = [3.5]
        alpha = 0.05
        distro = 'alpha'
        results = [True for _ in range(4) if KSTest(st.alpha.rvs(*parms, size=1000), distro, parms=parms, alpha=alpha,
                                                    display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in alpha GOF")

    def test_122_Kolmogorov_Smirnov_beta_test(self):
        """Test the beta distribution detection"""
        parms = [2.3, 0.6]
        alpha = 0.05
        distro = 'beta'
        results = [True for _ in range(4) if KSTest(st.beta.rvs(*parms, size=1000), distro, parms=parms, alpha=alpha,
                                                    display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in beta GOF")

    def test_123_Kolmogorov_Smirnov_cauchy_test(self):
        """Test the cauchy distribution detection"""
        alpha = 0.05
        distro = 'cauchy'
        results = [True for _ in range(4) if KSTest(st.cauchy.rvs(size=1000), distro, alpha=alpha,
                                                    display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in cauchy GOF")

    def test_124_Kolmogorov_Smirnov_chi2_large_test(self):
        """Test the chi squared distribution detection with sufficiently large dof"""
        parms = [50]
        alpha = 0.05
        distro = 'chi2'
        results = [True for _ in range(4) if KSTest(st.chi2.rvs(*parms, size=1000), distro, parms=parms,
                                                    alpha=alpha, display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in chi2 large GOF")

    def test_124_Kolmogorov_Smirnov_chi2_small_test(self):
        """Test the chi squared distribution detection with small dof"""
        parms = [5]
        alpha = 0.05
        distro = 'chi2'
        results = [True for _ in range(4) if KSTest(st.chi2.rvs(*parms, size=1000), distro, parms=parms,
                                                    alpha=alpha, display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in chi2 small GOF")

    def test_124_Kolmogorov_Smirnov_weibull_min_test(self):
        """Test the weibull min distribution detection"""
        parms = [1.7]
        alpha = 0.05
        distro = 'weibull_min'
        results = [True for _ in range(4) if KSTest(st.weibull_min.rvs(*parms, size=1000), distro, parms=parms,
                                                    alpha=alpha, display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in chi2 small GOF")

    def test_124_Kolmogorov_Smirnov_weibull_max_test(self):
        """Test the weibull min distribution detection"""
        parms = [2.8]
        alpha = 0.05
        distro = 'weibull_max'
        results = [True for _ in range(4) if KSTest(st.weibull_max.rvs(*parms, size=1000), distro, parms=parms,
                                                    alpha=alpha, display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Error in chi2 small GOF")

    def test_125_Norm_test(self):
        """Test the normal distribution check"""
        parms = [5, 0.1]
        alpha = 0.05
        results = [True for _ in range(4) if NormTest(st.norm.rvs(*parms, size=1000), display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Normal test Type I error")

    def test_126_Norm_test_fail(self):
        """Test the normal distribution check fails for a different distribution"""
        parms = [1.7]
        alpha = 0.05
        results = [True for _ in range(4) if NormTest(st.weibull_min.rvs(*parms, size=1000),
                                                      display=False).results[0] > alpha]
        self.assertFalse(True if True in results else False, "FAIL: Normal test Type II error")

    def test_127_TTest_equal_variance_matched(self):
        """Test the TTest with two samples with equal variance and matched means"""
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        alpha = 0.05
        results = [True for _ in range(4) if TTest(st.norm.rvs(*x_parms, size=1000), st.norm.rvs(*y_parms, size=1000),
                                                   display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: TTest equal variance matched Type I error")

    def test_128_TTest_equal_variance_unmatched(self):
        """Test the TTest with two samples with equal variance and different means"""
        x_parms = [4, 0.75]
        y_parms = [4.5, 0.75]
        alpha = 0.05
        results = [True for _ in range(4) if TTest(st.norm.rvs(*x_parms, size=1000), st.norm.rvs(*y_parms, size=1000),
                                                   display=False).results[0] > alpha]
        self.assertFalse(True if True in results else False, "FAIL: TTest equal variance unmatched Type II error")

    def test_129_TTest_unequal_variance_matched(self):
        """Test the TTest with two samples with different variances and matched means"""
        x_parms = [4, 0.75]
        y_parms = [4, 1.35]
        alpha = 0.05
        results = [True for _ in range(4) if TTest(st.norm.rvs(*x_parms, size=1000), st.norm.rvs(*y_parms, size=1000),
                                                   display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: TTest different variance matched Type I error")

    def test_130_TTest_unequal_variance_unmatched(self):
        """Test the TTest with two samples with different variances and different means"""
        x_parms = [4.0, 0.75]
        y_parms = [4.5, 1.12]
        alpha = 0.05
        results = [True for _ in range(4) if TTest(st.norm.rvs(*x_parms, size=1000), st.norm.rvs(*y_parms, size=1000),
                                                   display=False).results[0] > alpha]
        self.assertFalse(True if True in results else False, "FAIL: TTest different variance unmatched Type II error")

    def test_131_LinRegress_corr(self):
        """Test the Linear Regression class for correlation"""
        x_input_array = range(1, 101)
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertFalse(LinearRegression(x_input_array, y_input_array, alpha=alpha,
                                          display=False).results[0] > alpha, "FAIL: Linear Regression Type II error")

    def test_132_LinRegress_no_corr(self):
        """Test the Linear Regression class for uncorrelated data"""
        alpha = 0.05
        x_input_array = np.random.randn(200)
        y_input_array = np.random.randn(200)
        self.assertTrue(LinearRegression(x_input_array, y_input_array,
                                         display=False).results[0] > alpha, "FAIL: Linear Regression Type I error")

    def test_133_Correlation_corr_pearson(self):
        """Test the Correlation class for correlated normally distributed data"""
        x_input_array = list(np.random.randn(200))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertFalse(Correlation(x_input_array, y_input_array, alpha=alpha,
                                     display=False).results[0] > alpha, "FAIL: Correlation pearson Type II error")

    def test_134_Correlation_no_corr_pearson(self):
        """Test the Correlation class for uncorrelated normally distributed data"""
        alpha = 0.05
        results = [True for _ in range(4) if Correlation(np.random.randn(1000), np.random.randn(1000),
                                                         display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Correlation pearson Type I error")

    def test_135_Correlation_corr_spearman(self):
        """Test the Correlation class for correlated randomly distributed data"""
        x_input_array = list(np.random.rand(100))
        y_input_array = [x * 3 for x in x_input_array]
        alpha = 0.05
        self.assertFalse(Correlation(x_input_array, y_input_array, alpha=alpha,
                                     display=False).results[0] > alpha, "FAIL: Correlation spearman Type II error")

    def test_136_Correlation_no_corr_spearman(self):
        """Test the Correlation class for uncorrelated randomly distributed data"""
        x_input_array = np.random.rand(100)
        y_input_array = np.random.rand(100)
        alpha = 0.05
        self.assertTrue(Correlation(x_input_array, y_input_array, alpha=alpha,
                                    display=False).results[0] > alpha, "FAIL: Correlation spearman Type I error")

    def test_137_EqualVariance_Bartlett_matched(self):
        """Test the EqualVariance class for normally distributed matched variances"""
        x_parms = [4, 0.75]
        y_parms = [4, 0.75]
        z_parms = [4, 0.75]
        alpha = 0.05
        results = [True for _ in range(4) if EqualVariance(st.norm.rvs(*x_parms, size=1000),
                                                           st.norm.rvs(*y_parms, size=1000),
                                                           st.norm.rvs(*z_parms, size=1000),
                                                           display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Equal variance bartlett Type I error")

    def test_138_EqualVariance_Bartlett_unmatched(self):
        """Test the EqualVariance class for normally distributed unmatched variances"""
        x_parms = [4, 1.35]
        y_parms = [4, 1.35]
        z_parms = [4, 0.75]
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array = st.norm.rvs(*y_parms, size=1000)
        z_input_array = st.norm.rvs(*z_parms, size=1000)
        alpha = 0.05
        self.assertFalse(EqualVariance(x_input_array, y_input_array, z_input_array, alpha=alpha,
                                       display=False).results[0] > alpha, "FAIL: Equal variance bartlett Type II error")

    def test_139_EqualVariance_Levine_matched(self):
        """Test the EqualVariance class for non-normally distributed matched variances"""
        x_parms = [1.7]
        y_parms = [1.7]
        z_parms = [1.7]
        alpha = 0.05
        results = [True for _ in range(4) if EqualVariance(st.weibull_min.rvs(*x_parms, size=1000),
                                                           st.weibull_min.rvs(*y_parms, size=1000),
                                                           st.weibull_min.rvs(*z_parms, size=1000),
                                                           display=False).results[0] > alpha]
        self.assertTrue(True if True in results else False, "FAIL: Unequal variance levine Type I error")

    def test_140_EqualVariance_Levine_unmatched(self):
        """Test the EqualVariance class for non-normally distributed unmatched variances"""
        x_parms = [1.7]
        y_parms = [4, 0.75]
        z_parms = [1.7]
        alpha = 0.05
        results = [True for _ in range(4) if EqualVariance(st.weibull_min.rvs(*x_parms, size=1000),
                                                           st.norm.rvs(*y_parms, size=1000),
                                                           st.weibull_min.rvs(*z_parms, size=1000),
                                                           display=False).results[0] > alpha]
        self.assertFalse(True if True in results else False, "FAIL: Unequal variance levine Type II error")

    def test_141_Kruskal_matched(self):
        """Test the Kruskal Wallis class on matched data"""
        x_parms = [1.7]
        x_input_array = st.weibull_min.rvs(*x_parms, size=1000)
        y_input_array = st.weibull_min.rvs(*x_parms, size=1000)
        z_input_array = st.weibull_min.rvs(*x_parms, size=1000)
        alpha = 0.05
        self.assertTrue(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha,
                                display=True).results[0] > alpha, "FAIL: Kruskal Type I error")

    def test_142_Kruskal_unmatched(self):
        """Test the Kruskal Wallis class on unmatched data"""
        x_parms = [1.7]
        z_parms = [1.1]
        x_input_array = st.weibull_min.rvs(*x_parms, size=1000)
        y_input_array = st.weibull_min.rvs(*x_parms, size=1000)
        z_input_array = st.norm.rvs(*z_parms, size=1000)
        alpha = 0.05
        self.assertFalse(Kruskal(x_input_array, y_input_array, z_input_array, alpha=alpha,
                                 display=True).results[0] > alpha, "FAIL: Kruskal Type II error")

    def test_143_ANOVA_matched(self):
        """Test the ANOVA class on matched data"""
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array = st.norm.rvs(*x_parms, size=1000)
        z_input_array = st.norm.rvs(*x_parms, size=1000)
        alpha = 0.05
        self.assertTrue(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha,
                              display=True).results[0] > alpha, "FAIL: ANOVA Type I error")

    def test_144_ANOVA_unmatched(self):
        """Test the ANOVA class on unmatched data"""
        x_parms = [4, 1.75]
        y_parms = [6, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array =st.norm.rvs(*y_parms, size=1000)
        z_input_array = st.norm.rvs(*x_parms, size=1000)
        alpha = 0.05
        self.assertFalse(Anova(x_input_array, y_input_array, z_input_array, alpha=alpha,
                               display=True).results[0] > alpha, "FAIL: ANOVA Type II error")

    def test_145_GroupNorm_normal(self):
        """Test the GroupNorm class on normal data"""
        x_parms = [4, 1.75]
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array = st.norm.rvs(*x_parms, size=1000)
        z_input_array = st.norm.rvs(*x_parms, size=1000)
        alpha = 0.05
        self.assertTrue(GroupNormTest(x_input_array, y_input_array, z_input_array, alpha=alpha,
                                      display=True).results[0] > alpha, "FAIL: Group Norm Type I error")

    def test_146_GroupNorm_non_normal(self):
        """Test the GroupNorm class on non-normal data"""
        x_parms = [4, 1.75]
        z_parms = [1.7]
        x_input_array = st.norm.rvs(*x_parms, size=1000)
        y_input_array = st.norm.rvs(*x_parms, size=1000)
        z_input_array = st.weibull_min.rvs(*z_parms, size=1000)
        alpha = 0.05
        self.assertFalse(GroupNormTest(x_input_array, y_input_array, z_input_array, alpha=alpha,
                                       display=True).results[0] > alpha, "FAIL: Group Norm Type II error")

    def test_147_Vector_stats(self):
        """Test the vector statistics class"""
        parms = [4, 1.75]
        comp = [100, parms[0], parms[1]]
        input_array = st.norm.rvs(*parms, size=comp[0])
        results = VectorStatistics(input_array, sample=True, display=True).results
        test = (results['count'], results['mean'], results['std'])
        check = [abs(comp[i] - test[i]) for i in range(3)]
        self.assertTrue(check[0] < 0.5 and check[1] < 0.5 and check[2] < 0.5, "FAIL: Stat delta is too large")


if __name__ == '__main__':
    unittest.main()