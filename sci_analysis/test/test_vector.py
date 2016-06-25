import unittest
import numpy as np
from ..data.vector import Vector
from ..operations.data_operations import drop_nan, drop_nan_intersect


class MyTestCase(unittest.TestCase):
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

    def test_123_vector_is_empty_none(self):
        """Test the vector is_empty method"""
        input_array = None
        self.assertTrue(Vector(input_array).is_empty(), "FAIL: Error vector is_empty")

        # Test drop nan functions

    def test_124_drop_nan(self):
        """Test the drop_nan method"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = np.array([1.0, 3.0, 4.1])
        self.assertTrue(np.array_equal(drop_nan(Vector(input_array)), out_array), "FAIL: Error in drop_nan")

    def test_125_drop_nan_intersect(self):
        """Test the drop_nan_intersect method"""
        input_array_1 = [1., float("nan"), 3., float("nan"), 5.]
        input_array_2 = [11., float("nan"), 13., 14., 15.]
        output_array = [(1., 11.), (3., 13.), (5., 15.)]
        inter = drop_nan_intersect(Vector(input_array_1), Vector(input_array_2))
        self.assertTrue(zip(inter[0], inter[1]) == output_array, "FAIL: Error in drop_nan_intersect")

    def test_126_drop_nan_empty(self):
        """Test the drop_nan method on an empty array"""
        input_array = np.array([])
        self.assertFalse(drop_nan(input_array), "FAIL: Error in drop_nan empty array")


if __name__ == '__main__':
    unittest.main()
