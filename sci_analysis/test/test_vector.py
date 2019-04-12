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
        self.assertTrue(all(pd.isna(test_obj._values['lbl'])))
        self.assertEqual(['None'] * 100, test_obj.labels.tolist())

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
        """Test that paired groups doesn't return empty groups.."""
        ind_x_1 = [0, 1, 2, 3, 4]
        ind_y_1 = [5, 6, 7, 8, 9]
        ind_x_2 = [10, 11, 12, 13, 14]
        ind_y_2 = [15, 16, 17, 18, 19]
        input1 = Vector(ind_x_1, other=ind_y_1)
        input2 = Vector(ind_x_2, other=ind_y_2)
        new_input = input1.append(Vector(pd.Series([]))).append(input2)
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
        """Test that a ValueError is raised when the input array is a pandas DataFrame."""
        input_array = pd.DataFrame([1, 2, 3], [4, 5, 6])
        self.assertRaises(ValueError, lambda: Vector(input_array))

    def test_154_vector_with_labels(self):
        """Test that labels are created correctly and available with the labels property."""
        np.random.seed(987654321)
        input_array = st.norm.rvs(size=100)
        labels = np.random.randint(10000, 50000, size=100)
        test_obj = Vector(input_array, labels=labels)
        self.assertListEqual(pd.Series(labels).tolist(), test_obj.labels.tolist())
        self.assertIsInstance(test_obj.labels, pd.Series)

    def test_155_vector_drop_nan_with_labels(self):
        """Test to make sure labels are properly dropped when drop_nan is called."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        labels = np.random.randint(10000, 50000, size=100)
        input_array_1[8] = np.nan
        input_array_1[16] = np.nan
        input_array_1[32] = np.nan
        input_array_1[64] = np.nan
        input_array_1[17] = np.nan
        input_array_1[22] = np.nan
        input_array_1[43] = np.nan
        input_array_1[89] = np.nan
        test_obj = Vector(input_array_1, labels=labels)
        self.assertEqual(len(test_obj.labels), 92)
        self.assertRaises(KeyError, lambda: test_obj.labels[32])

    def test_156_vector_drop_nan_intersect_with_labels(self):
        """Test to make sure labels are properly dropped when drop_nan_intersect is called."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        labels = np.random.randint(10000, 50000, size=100)
        input_array_1[8] = np.nan
        input_array_1[16] = np.nan
        input_array_1[32] = np.nan
        input_array_1[64] = np.nan
        input_array_2[1] = np.nan
        input_array_2[2] = np.nan
        input_array_2[4] = np.nan
        input_array_2[8] = np.nan
        test_obj = Vector(input_array_1, input_array_2, labels=labels)
        self.assertEqual(len(test_obj.labels), 93)
        self.assertRaises(KeyError, lambda: test_obj.labels[32])
        self.assertRaises(KeyError, lambda: test_obj.labels[8])

    def test_157_vector_labels_single_value(self):
        """Test that if a single value is passed in to labels, the value is applied to all rows."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        labels = 42
        test_obj = Vector(input_array_1, input_array_2, labels=labels)
        self.assertListEqual([42] * 100, test_obj.labels.tolist())

    def test_158_vector_label_as_None(self):
        """Test that missing label values are converted to the string 'None'."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        labels = np.random.randint(10000, 50000, size=100).astype('str')
        input_array_1[8] = np.nan
        input_array_1[16] = np.nan
        input_array_1[32] = np.nan
        input_array_1[64] = np.nan
        input_array_2[1] = np.nan
        input_array_2[2] = np.nan
        input_array_2[4] = np.nan
        input_array_2[8] = np.nan
        labels[24] = None
        labels[48] = None
        labels[72] = None
        labels[96] = None
        test_obj = Vector(input_array_1, input_array_2, labels=labels)
        self.assertEqual(len(test_obj.labels), 93)
        self.assertEqual('None', test_obj.labels[24])

    def test_159_vector_unequal_labels_length(self):
        """Test to make sure that an error is raised if the length of labels is unequal to the input data."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        labels = np.random.randint(10000, 50000, size=50)
        self.assertRaises(UnequalVectorLengthError, lambda: Vector(input_array_1, input_array_2, labels=labels))

    def test_160_vector_groups_with_labels(self):
        """Test to make sure group_labels returns the expected output."""
        ind_x_1 = [0, 1, 2, 3, 4]
        ind_y_1 = [5, 6, 7, 8, 9]
        ind_x_2 = [10, 11, 12, 13, 14]
        ind_y_2 = [15, 16, 17, 18, 19]
        labels_1 = ['A', 'B', 'C', 'D', 'E']
        labels_2 = ['AA', 'BB', 'CC', 'DD', 'EE']
        input1 = Vector(ind_x_1, other=ind_y_1, labels=labels_1)
        input2 = Vector(ind_x_2, other=ind_y_2, labels=labels_2)
        new_input = input1.append(Vector(pd.Series([]))).append(input2)
        groups = new_input.group_labels
        self.assertDictEqual({1: labels_1, 2: labels_2}, {grp: l.tolist() for grp, l in groups.items()})
        self.assertListEqual([1, 2], list(groups.keys()))

    def test_161_vector_has_labels(self):
        """Test to verify the logic for the has_labels property is working as expected."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        labels = np.random.randint(10000, 50000, size=100).astype(str)
        self.assertTrue(Vector(input_array_1, labels=labels).has_labels)
        self.assertFalse(Vector(input_array_1).has_labels)
        labels[5] = None
        labels[10] = None
        self.assertTrue(Vector(input_array_1, labels=labels).has_labels)
        labels = [None] * 100
        labels[5] = 'hi'
        self.assertTrue(Vector(input_array_1, labels=labels).has_labels)

    def test_162_vector_drop_group(self):
        """Test the normal use case for dropping a group from the Vector."""
        np.random.seed(987654321)
        input_array_1 = st.norm.rvs(size=100)
        input_array_2 = st.norm.rvs(size=100)
        input_array_3 = st.norm.rvs(size=100)
        input_array_4 = st.norm.rvs(size=4)
        vec1 = (
            Vector(input_array_1)
            .append(Vector(input_array_2))
            .append((Vector(input_array_3)))
            .append(Vector(input_array_4))
        )
        self.assertEqual(len(vec1.drop_groups(4)), 300)
        self.assertEqual(len(vec1.drop_groups(2)), 200)
        self.assertListEqual([1, 3], vec1.values['grp'].cat.categories.tolist())
        vec1_1 = (
            Vector(input_array_1)
                .append(Vector(input_array_2))
                .append((Vector(input_array_3)))
                .append(Vector(input_array_4))
        )
        self.assertEqual(len(vec1_1.drop_groups([2, 4])), 200)
        self.assertListEqual([1, 3], vec1_1.values['grp'].cat.categories.tolist())
        vec2 = (
            Vector(input_array_1, groups=['a'] * 100)
            .append(Vector(input_array_2, groups=['b'] * 100))
            .append((Vector(input_array_3, groups=['c'] * 100)))
            .append(Vector(input_array_4, groups=['d'] * 4))
        )
        self.assertEqual(len(vec2.drop_groups('b')), 204)
        self.assertEqual(len(vec2.drop_groups('d')), 200)
        self.assertListEqual(['a', 'c'], vec2.values['grp'].cat.categories.tolist())
        vec2_1 = (
            Vector(input_array_1, groups=['a'] * 100)
                .append(Vector(input_array_2, groups=['b'] * 100))
                .append((Vector(input_array_3, groups=['c'] * 100)))
                .append(Vector(input_array_4, groups=['d'] * 4))
        )
        self.assertEqual(len(vec2_1.drop_groups(['b', 'd'])), 200)
        self.assertListEqual(['a', 'c'], vec2_1.values['grp'].cat.categories.tolist())


if __name__ == '__main__':
    unittest.main()
