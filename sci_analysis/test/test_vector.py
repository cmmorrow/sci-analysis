import unittest
import numpy as np
import scipy.stats as st
from data.data import Vector, UnequalVectorLengthError


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
        self.assertFalse(Vector().data, "FAIL: Error in empty list vector creation")

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

    def test_106_create_vector_array(self):
        """Test vector creation from an array"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100)
        self.assertEqual(Vector(input_array).data_type, np.dtype('float64'),
                         "FAIL: Error in array vector creation dtype")

    def test_107_create_vector_array_large(self):
        """Test vector creation from a large array"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=1000000)
        self.assertEqual(Vector(input_array).data_type, np.dtype('float64'),
                         "FAIL: Error in large array vector creation dtype")

    def test_108_create_vector_from_vector(self):
        """Test vector creation from a vector"""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=100))
        self.assertEqual(Vector(input_array).data_type, np.dtype('float64'),
                         "FAIL: Error in vector from vector creation dtype")

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
        self.assertTrue(np.array_equal(Vector(input_array).drop_nan(), out_array), "FAIL: Error in drop_nan")

    def test_125_drop_nan_empty(self):
        """Test the drop_nan method on an empty array"""
        input_array = ["one", "two", "three", "four"]
        self.assertFalse(Vector(input_array).drop_nan(), "FAIL: drop_nan did not identify the empty array")

    def test_126_drop_nan_intersect(self):
        """Test the drop_nan_intersect method"""
        input_array_1 = [1., float("nan"), 3., float("nan"), 5.]
        input_array_2 = [11., float("nan"), 13., 14., 15.]
        # output_array = [(1., 11.), (3., 13.), (5., 15.)]
        output_array = (np.array([1., 3., 5.]), np.array([11., 13., 15.]))
        inter = Vector(input_array_1).drop_nan_intersect(Vector(input_array_2))
        test1 = inter[0] == output_array[0]
        test2 = inter[1] == output_array[1]
        # self.assertEqual(zip(inter[0], inter[1]), output_array, "FAIL: Error in drop_nan_intersect")
        self.assertTrue(test1.all and test2.all, "FAIL: Error in drop_nan_intersect")

    def test_127_drop_nan_intersect_empty(self):
        """Test the drop_nan_intersect method with one empty array"""
        input_array_2 = ["one", "two", "three", "four", "five"]
        input_array_1 = [11., float("nan"), 13., 14., 15.]
        self.assertEqual(len(Vector(input_array_1).drop_nan_intersect(Vector(input_array_2))[0]), 0,
                         "FAIL: drop_nan_intersect did not identify the empty array")

    def test_128_drop_nan_empty(self):
        """Test the drop_nan method on an empty array"""
        input_array = Vector(np.array([]))
        self.assertFalse(input_array.drop_nan(), "FAIL: Error in drop_nan empty array")

    def test_129_vector_data_prep(self):
        """Test the vector data_prep method"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100)
        input_array[4] = float("nan")
        input_array[16] = float("nan")
        input_array[32] = float("nan")
        input_array[64] = float("nan")
        self.assertEqual(len(Vector(input_array).data_prep()), 96, "FAIL: Error in array data prep")

    def test_130_vector_data_prep_empty(self):
        """Test the vector data_prep method when the vector is empty"""
        input_array = np.array([])
        self.assertFalse(Vector(input_array).data_prep(), "FAIL: data_prep did not return None")

    def test_131_vector_data_prep_two_arrays(self):
        """Test the vector data_prep method when there are two vectors"""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        input_array_1[8] = float("nan")
        input_array_1[16] = float("nan")
        input_array_1[32] = float("nan")
        input_array_1[64] = float("nan")
        input_array_2[1] = float("nan")
        input_array_2[2] = float("nan")
        input_array_2[4] = float("nan")
        input_array_2[8] = float("nan")
        result = Vector(input_array_1).data_prep(Vector(input_array_2))
        self.assertEqual((len(result[0]), len(result[1])), (93, 93),
                         "FAIL: Error in data prep with two vectors")

    def test_132_vector_data_prep_two_unequal_arrays(self):
        """Test the vector data_prep method when there are two vectors with different lengths"""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=92)
        input_array_2 = st.norm.rvs(size=100)
        self.assertRaises(UnequalVectorLengthError, lambda: Vector(input_array_1).data_prep(Vector(input_array_2)))

    def test_133_vector_data_prep_two_empty_arrays(self):
        """Test the vector data_prep method when there are two empty vectors"""
        input_array_1 = ["one", "two", "three", "four", "five"]
        input_array_2 = ["three", "four", "five", "six", "seven"]
        # self.assertRaises(EmptyVectorError, lambda: Vector(input_array_1).data_prep(Vector(input_array_2)))
        self.assertFalse(Vector(input_array_1).data_prep(Vector(input_array_2)),
                         "FAIL: data_prep did not return None")

    def test_134_vector_data_prep_int(self):
        """Test the vector data_prep method on an int value"""
        self.assertEqual(Vector(4).data_prep(), np.array([4.]), "FAIL: Error data prep on float value")

    def test_135_vector_data_prep_float(self):
        """Test the vector data_prep method on an int value"""
        self.assertEqual(Vector(4.0).data_prep(), np.array([4.]), "FAIL: Error data prep on float value")

    def test_136_vector_data_prep_string(self):
        """Test the vector data_prep method on an int value"""
        # self.assertRaises(EmptyVectorError, lambda: Vector("four").data_prep())
        self.assertFalse(Vector("four").data_prep(), "FAIL: data_prep did not return None")

    def test_137_vector_data_prep_no_nan(self):
        """Test the vector data_prep method on a vector with no nan values"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100)
        self.assertGreater(len(Vector(input_array).data_prep()), 1, "FAIL: Error data_prep normal array")


if __name__ == '__main__':
    unittest.main()
