import unittest
import numpy as np
import pandas as pd
import scipy.stats as st
from ..data import Vector, UnequalVectorLengthError


class MyTestCase(unittest.TestCase):
    # Test vector creation

    def test_100_create_vector_mixed_list(self):
        """Test vector creation from a mixed list"""
        input_array = [1.0, "2", '3.0', "four", 5.65]
        out_array = [1.0, 2.0, 3.0, 5.65]
        self.assertListEqual(out_array, Vector(input_array).data.tolist())

    def test_101_create_vector_missing_val(self):
        """Test vector creation from a missing value list"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = [1.0, 3.0, 4.1]
        self.assertListEqual(out_array, Vector(input_array).data.tolist())

    def test_102_create_vector_empty_list(self):
        """Test vector creation from an empty list"""
        self.assertTrue(Vector().data.empty)

    def test_103_create_vector_2dim_array(self):
        """Test vector creation from a 2dim array"""
        input_array = np.array([[1, 2, 3], [1, 2, 3]])
        out_array = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        self.assertListEqual(out_array, Vector(input_array).data.tolist())

    def test_104_create_vector_dict(self):
        """Test vector creation from a dict"""
        input_array = {"one": 1, "two": 2.0, "three": "3", "four": "four"}
        self.assertTrue(Vector(input_array).is_empty())

    def test_105_create_vector_tuple(self):
        """Test vector creation from a tuple"""
        input_array = (1, 2, 3, 4, 5)
        out_array = [1., 2., 3., 4., 5.]
        self.assertListEqual(out_array, Vector(input_array).data.tolist())

    def test_106_create_vector_array(self):
        """Test vector creation from an array"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100)
        test_obj = Vector(input_array)
        self.assertEqual(len(test_obj), 100)
        self.assertIsInstance(test_obj, Vector)
        self.assertIsInstance(test_obj.data, pd.Series)
        self.assertEqual(test_obj.data_type, np.dtype('float64'))

    def test_107_create_vector_array_large(self):
        """Test vector creation from a large array"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=1000000)
        test_obj = Vector(input_array)
        self.assertEqual(len(test_obj), 1000000)
        self.assertIsInstance(test_obj, Vector)
        self.assertIsInstance(test_obj.data, pd.Series)
        self.assertEqual(test_obj.data_type, np.dtype('float64'))

    def test_108_create_vector_from_vector(self):
        """Test vector creation from a vector"""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=100))
        second_array = Vector(input_array)
        # print(id(input_array))
        # print(id(second_array))
        self.assertEqual(second_array.data_type, np.dtype('float64'))

    def test_109_create_vector_2dim_list(self):
        """Test vector creation from a 2dim list"""
        input_array = [[1, 2, 3], [1, 2, 3]]
        out_array = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        self.assertListEqual(out_array, Vector(input_array).data.tolist())

    def test_120_create_vector_none(self):
        """Test vector creation from None"""
        self.assertTrue(Vector(None).is_empty())

    def test_121_vector_is_empty_empty_list(self):
        """Test the vector is_empty method"""
        input_array = []
        self.assertTrue(Vector(input_array).is_empty())

    def test_122_vector_is_empty_empty_array(self):
        """Test the vector is_empty method"""
        input_array = np.array([])
        self.assertTrue(Vector(input_array).is_empty())

        # Test drop nan functions

    def test_124_drop_nan(self):
        """Test the drop_nan method"""
        input_array = ["1.0", "", 3, '4.1', ""]
        out_array = [1.0, 3.0, 4.1]
        self.assertListEqual(out_array, Vector(input_array).data.tolist())

    def test_125_drop_nan_empty(self):
        """Test the drop_nan method on an empty array"""
        input_array = ["one", "two", "three", "four"]
        self.assertTrue(Vector(input_array).is_empty())

    def test_126_drop_nan_intersect(self):
        """Test the drop_nan_intersect method"""
        input_array_1 = [1., np.nan, 3., np.nan, 5.]
        input_array_2 = [11., np.nan, 13., 14., 15.]
        out1 = [1., 3., 5.]
        out2 = [11., 13., 15.]
        vector = Vector(input_array_1, input_array_2)
        self.assertListEqual(out1, vector.data.tolist())
        self.assertListEqual(out2, vector.other.tolist())

    def test_127_drop_nan_intersect_empty(self):
        """Test the drop_nan_intersect method with one empty array"""
        # This test caught a bug when developing the Vector constructor refactor in 2.0.0
        input_array_2 = ["one", "two", "three", "four", "five"]
        input_array_1 = [11., np.nan, 13., 14., 15.]
        self.assertTrue(Vector(input_array_1, input_array_2).other.empty)

    def test_129_vector_data_prep(self):
        """Test the vector data_prep method"""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100)
        input_array[4] = np.nan
        input_array[16] = np.nan
        input_array[32] = np.nan
        input_array[64] = np.nan
        self.assertEqual(len(Vector(input_array)), 96)

    def test_131_vector_data_prep_two_arrays(self):
        """Test the vector data_prep method when there are two vectors"""
        # This test caught a bug when developing the Vector constructor refactor in 2.0.0
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        input_array_1[8] = np.nan
        input_array_1[16] = np.nan
        input_array_1[32] = np.nan
        input_array_1[64] = np.nan
        input_array_2[1] = np.nan
        input_array_2[2] = np.nan
        input_array_2[4] = np.nan
        input_array_2[8] = np.nan
        vector = Vector(input_array_1, input_array_2)
        x, y = vector.data, vector.other
        self.assertEqual((len(x), len(y)), (93, 93))

    def test_132_vector_data_prep_two_unequal_arrays(self):
        """Test the vector data_prep method when there are two vectors with different lengths"""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=92)
        input_array_2 = st.norm.rvs(size=100)
        self.assertRaises(UnequalVectorLengthError, lambda: Vector(input_array_1, input_array_2))

    def test_133_vector_data_prep_two_empty_arrays(self):
        """Test the vector data_prep method when there are two empty vectors"""
        input_array_1 = ["one", "two", "three", "four", "five"]
        input_array_2 = ["three", "four", "five", "six", "seven"]
        self.assertTrue(Vector(input_array_1, input_array_2).is_empty())

    def test_134_vector_data_prep_int(self):
        """Test the vector data_prep method on an int value"""
        self.assertTrue(Vector(4).data.equals(pd.Series([4.], name='ind')))

    def test_135_vector_data_prep_float(self):
        """Test the vector data_prep method on an int value"""
        self.assertTrue(Vector(4.0).data.equals(pd.Series([4.], name='ind')))

    def test_136_vector_data_prep_string(self):
        """Test the vector data_prep method on an int value"""
        self.assertTrue(Vector("four").is_empty())

    def test_137_basic_groupby(self):
        """Test the group property produces the correct dictionary"""
        ind = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        grp = ['a', 'b', 'c', 'c', 'a', 'b', 'b', 'c', 'a']
        groups = Vector(ind, groups=grp).groups
        self.assertTrue(groups['a'].equals(pd.Series([1., 2., 3.], index=[0, 4, 8], name='a')))
        self.assertTrue(groups['b'].equals(pd.Series([2., 3., 1.], index=[1, 5, 6], name='b')))
        self.assertTrue(groups['c'].equals(pd.Series([3., 1., 2.], index=[2, 3, 7], name='c')))

    def test_138_nan_groupby(self):
        """Test the group property where certain values in data are NaN."""
        ind = [1, np.nan, 3, np.nan, 2, 3, np.nan, 2, 3]
        grp = ['a', 'b', 'c', 'c', 'a', 'b', 'b', 'c', 'a']
        groups = Vector(ind, groups=grp).groups
        self.assertTrue(groups['a'].equals(pd.Series([1., 2., 3.], index=[0, 4, 8], name='a')))
        self.assertTrue(groups['b'].equals(pd.Series([3.], index=[5], name='b')))
        self.assertTrue(groups['c'].equals(pd.Series([3., 2.], index=[2, 7], name='c')))

    def test_139_nan_drop_groupby(self):
        """Test the group property where certain values in data are NaN which causes a group to be dropped."""
        ind = [1, np.nan, 3, 1, 2, np.nan, np.nan, 2, 3]
        grp = ['a', 'b', 'c', 'c', 'a', 'b', 'b', 'c', 'a']
        groups = Vector(ind, groups=grp).groups
        self.assertTrue(groups['a'].equals(pd.Series([1., 2., 3.], index=[0, 4, 8], name='a')))
        self.assertTrue(groups['c'].equals(pd.Series([3., 1., 2.], index=[2, 3, 7], name='c')))
        self.assertNotIn('b', groups.keys())

    def test_140_vector_groups_dtype_passed_group_names(self):
        """Test to make sure the dtype of the groups column is categorical."""
        ind = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        grp = ['a', 'b', 'c', 'c', 'a', 'b', 'b', 'c', 'a']
        groups = Vector(ind, groups=grp)
        self.assertEqual(groups.values['grp'].dtype, 'category')

    def test_141_vector_groups_dtype_passed_no_group(self):
        """Test to make sure the dtype of the groups column is categorical."""
        ind = st.norm.rvs(size=1000)
        groups = Vector(ind)
        self.assertEqual(groups.values['grp'].dtype, 'category')

    def test_142_vector_append_existing_groups_with_new_groups(self):
        """Test appending a new vector to an existing one."""
        ind1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        grp1 = ['a', 'b', 'c', 'c', 'a', 'b', 'b', 'c', 'a']
        ind2 = [1, 2, 3]
        grp2 = ['d', 'd', 'd']
        input1 = Vector(ind1, groups=grp1)
        input2 = Vector(ind2, groups=grp2)
        new_input = input1.append(input2)
        groups = new_input.groups
        self.assertTrue(groups['a'].equals(pd.Series([1., 2., 3.], index=[0, 4, 8], name='a')))
        self.assertTrue(groups['b'].equals(pd.Series([2., 3., 1.], index=[1, 5, 6], name='b')))
        self.assertTrue(groups['c'].equals(pd.Series([3., 1., 2.], index=[2, 3, 7], name='c')))
        self.assertTrue(groups['d'].equals(pd.Series([1., 2., 3.], index=[9, 10, 11], name='d')))
        self.assertIn('d', groups.keys())

    def test_143_vector_append_existing_groups_with_existing_groups(self):
        """Test appending a new vector to an existing one."""
        ind1 = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        grp1 = ['a', 'b', 'c', 'c', 'a', 'b', 'b', 'c', 'a']
        ind2 = [1, 2, 3]
        grp2 = ['a', 'b', 'c']
        input1 = Vector(ind1, groups=grp1)
        input2 = Vector(ind2, groups=grp2)
        new_input = input1.append(input2)
        groups = new_input.groups
        self.assertTrue(groups['a'].equals(pd.Series([1., 2., 3., 1.], index=[0, 4, 8, 9], name='a')))
        self.assertTrue(groups['b'].equals(pd.Series([2., 3., 1., 2.], index=[1, 5, 6, 10], name='b')))
        self.assertTrue(groups['c'].equals(pd.Series([3., 1., 2., 3.], index=[2, 3, 7, 11], name='c')))

    def test_144_vector_append_generated_groups_1(self):
        """Test appending a new vector to an existing one."""
        ind1 = [0, 1, 2, 3, 4]
        ind2 = [5, 6, 7, 8, 9]
        input1 = Vector(ind1)
        input2 = Vector(ind2)
        new_input = input1.append(input2)
        groups = new_input.groups
        self.assertTrue(groups[1].equals(pd.Series([0., 1., 2., 3., 4.], index=[0, 1, 2, 3, 4], name=1)))
        self.assertTrue(groups[2].equals(pd.Series([5., 6., 7., 8., 9.], index=[5, 6, 7, 8, 9], name=2)))
        self.assertListEqual([1, 2], list(groups.keys()))

    def test_145_vector_append_generated_groups_2(self):
        """Test appending a new vector to an existing one."""
        ind1 = [0, 1, 2, 3, 4]
        ind2 = [5, 6, 7, 8, 9]
        ind3 = [10, 11, 12, 13, 14]
        input1 = Vector(ind1)
        input2 = Vector(ind2)
        input3 = Vector(ind3)
        new_input = input1.append(input2).append(input3)
        groups = new_input.groups
        self.assertTrue(groups[1].equals(pd.Series([0., 1., 2., 3., 4.], index=[0, 1, 2, 3, 4], name=1)))
        self.assertTrue(groups[2].equals(pd.Series([5., 6., 7., 8., 9.], index=[5, 6, 7, 8, 9], name=2)))
        self.assertTrue(groups[3].equals(pd.Series([10., 11., 12., 13., 14.], index=[10, 11, 12, 13, 14], name=3)))
        self.assertListEqual([1, 2, 3], list(groups.keys()))

    def test_146_vector_append_not_a_vector(self):
        """Test the error raised by appending a non-vector object."""
        input1 = [1, 2, 3, 4, 5]
        input2 = [6, 7, 8, 9, 10]
        self.assertRaises(ValueError, lambda: Vector(input1).append(input2))

    def test_147_empty_vector_append_none(self):
        """Test to make sure appending an empty Vector returns the original Vector."""
        input_array = []
        self.assertTrue(Vector(input_array).append(Vector(None)).data.empty)

    def test_148_vector_append_none(self):
        """Test to make sure appending an empty Vector returns the original Vector."""
        input_array = [1, 2, 3, 4, 5]
        self.assertTrue(Vector(input_array).append(Vector(None)).data.equals(pd.Series(input_array).astype('float')))

    def test_149_vector_paired_groups(self):
        """Test paired groups."""
        ind_x_1 = [0, 1, 2, 3, 4]
        ind_y_1 = [5, 6, 7, 8, 9]
        ind_x_2 = [10, 11, 12, 13, 14]
        ind_y_2 = [15, 16, 17, 18, 19]
        input1 = Vector(ind_x_1, other=ind_y_1)
        input2 = Vector(ind_x_2, other=ind_y_2)
        new_input = input1.append(input2)
        groups = new_input.paired_groups
        self.assertTrue(groups[1][0].equals(pd.Series([0., 1., 2., 3., 4.])))
        self.assertTrue(groups[1][1].equals(pd.Series([5., 6., 7., 8., 9.])))
        self.assertTrue(groups[2][0].equals(pd.Series([10., 11., 12., 13., 14.], index=[5, 6, 7, 8, 9])))
        self.assertTrue(groups[2][1].equals(pd.Series([15., 16., 17., 18., 19.], index=[5, 6, 7, 8, 9])))
        self.assertListEqual([1, 2], list(groups.keys()))

    def test_150_vector_flatten_singled(self):
        """Test the Vector flatten method on a single vector."""
        np.random.seed(987654321)
        input_array = Vector(st.norm.rvs(size=100))
        self.assertEqual(len(input_array.flatten()), 1)
        self.assertTrue(input_array.data.equals(input_array.flatten()[0]))

    def test_151_vector_flatten_several_groups(self):
        """Test the Vector flatten method on a a single vector with multiple groups."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        input_array_3 = st.norm.rvs(size=100)
        input_array = Vector(input_array_1).append(Vector(input_array_2)).append(Vector(input_array_3))
        self.assertEqual(len(input_array.flatten()), 3)
        self.assertEqual(type(input_array.flatten()), tuple)
        self.assertTrue(input_array.groups[1].equals(input_array.flatten()[0]))
        self.assertTrue(input_array.groups[2].equals(input_array.flatten()[1]))
        self.assertTrue(input_array.groups[3].equals(input_array.flatten()[2]))

    def test_152_vector_flatten_several_paired_groups(self):
        """Test the Vector flatten method on a paired vector with multiple groups."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        input_array_3 = st.norm.rvs(size=100)
        input_array_4 = st.norm.rvs(size=100)
        input_array = Vector(input_array_1, other=input_array_2).append(Vector(input_array_3, other=input_array_4))
        self.assertEqual(len(input_array.flatten()), 4)
        self.assertTrue(input_array.groups[1].equals(input_array.flatten()[0]))
        self.assertTrue(input_array.groups[2].equals(input_array.flatten()[1]))
        self.assertTrue(input_array.paired_groups[1][1].equals(input_array.flatten()[2]))
        self.assertTrue(input_array.paired_groups[2][1].equals(input_array.flatten()[3]))

    def test_153_vector_data_frame(self):
        input_array = pd.DataFrame([1, 2, 3], [4, 5, 6])
        self.assertRaises(ValueError, lambda: Vector(input_array))


if __name__ == '__main__':
    unittest.main()
