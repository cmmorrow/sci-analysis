import unittest
from warnings import catch_warnings, simplefilter
import numpy as np
from pandas import Series, MultiIndex

from data import Categorical, is_categorical, is_data, NumberOfCategoriesWarning


class TestWarnings(unittest.TestCase):
    """A TestCase subclass with assertWarns substitute to cover python 2.7 which doesn't have an assertWarns method."""

    def assertWarnsCrossCompatible(self, expected_warning, *args, **kwargs):
        with catch_warnings(record=True) as warning_list:
            simplefilter('always')
            callable_obj = args[0]
            args = args[1:]
            callable_obj(*args, **kwargs)
            self.assertTrue(any(item.category == expected_warning for item in warning_list))


class MyTestCase(TestWarnings):

    def test_100_create_categorical_simple(self):
        """Create a simple categorical object"""
        input_array = Categorical(["a", "b", "c", "d", "b", "c", "b"])
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1, 'b': 3, 'c': 2, 'd': 1})
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd'])
        self.assertFalse(input_array.is_empty())

    def test_101_create_simple_ordered(self):
        """Create an ordered, simple categorical object"""
        input_array = Categorical(["a", "b", "c", "d", "b", "c", "b"], order=['d', 'c', 'b', 'a'])
        self.assertTrue(is_categorical(input_array))
        self.assertListEqual(input_array.order, ['d', 'c', 'b', 'a'])
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1, 'b': 3, 'c': 2, 'd': 1})
        self.assertListEqual(input_array.categories.tolist(), ['d', 'c', 'b', 'a'])
        self.assertFalse(input_array.is_empty())

    def test_102_create_categorical_copy(self):
        """Create a copy of a simple categorical object"""
        ref = Categorical(["a", "b", "c", "d", "b", "c", "b"], name='test')
        input_array = Categorical(ref)
        self.assertTrue(is_data(ref))
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertEqual(input_array.name, 'test')
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1, 'b': 3, 'c': 2, 'd': 1})
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd'])
        self.assertFalse(input_array.is_empty())

    def test_103_create_categorical_from_none(self):
        """Create an empty Categorical object from None"""
        input_array = Categorical(None)
        self.assertEqual(len(input_array.data), 0)
        self.assertEqual(len(input_array.counts), 0)
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertIsNone(input_array.categories)
        self.assertTrue(input_array.is_empty)

    def test_104_create_categorical_from_empty_array(self):
        """Create an empty Categorical object from an empty array"""
        input_array = Categorical(np.array([]))
        self.assertEqual(len(input_array.data), 0)
        self.assertEqual(len(input_array.counts), 0)
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertIsNone(input_array.categories)
        self.assertTrue(input_array.is_empty)

    def test_105_create_categorical_from_simple_array(self):
        """Create a Categorical object from a simple numpy array"""
        input_array = Categorical(np.array(["a", "b", "c", "d", "b", "c", "b"]))
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1, 'b': 3, 'c': 2, 'd': 1})
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd'])
        self.assertFalse(input_array.is_empty())

    def test_106_create_categorical_from_series(self):
        """Create a Categorical object from a simple Series"""
        input_array = Categorical(Series(["a", "b", "c", "d", "b", "c", "b"]))
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1, 'b': 3, 'c': 2, 'd': 1})
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd'])
        self.assertFalse(input_array.is_empty())

    def test_107_create_categorical_drop_missing_data(self):
        """Create a Categorical object containing missing values and drop them"""
        i = ["a", "b", "c", "d", np.nan, "b", "c", np.nan, "b"]
        ref = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 5: 'b', 6: 'c', 8: 'b'}
        input_array = Categorical(i, name='test', dropna=True)
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertEqual(input_array.name, 'test')
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1, 'b': 3, 'c': 2, 'd': 1})
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd'])
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series(ref).astype('category')))

    def test_108_create_categorical_with_single_value_list(self):
        """Create a Categorical object containing only a single value"""
        input_array = Categorical(['a'])
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 1})
        self.assertListEqual(input_array.categories.tolist(), ['a'])
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series(input_array).astype('category')))

    def test_109_create_categorical_from_a_string(self):
        """Create a Categorical object from a string"""
        input_array = Categorical('abcde')
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'abcde': 1})
        self.assertListEqual(input_array.categories.tolist(), ['abcde'])
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series(input_array).astype('category')))

    def test_110_create_categorical_from_numeric(self):
        """Create a Categorical object from a numpy array"""
        np.random.seed(987654321)
        input_array = Categorical(np.random.normal(size=10).astype('float32'))
        order = [-1.3409367799758911, -1.1835769414901733,
                 -0.99705976247787476, -0.64591825008392334,
                 0.066813990473747253, 0.70958340167999268,
                 0.81532984972000122, 1.9106369018554688,
                 1.9294924736022949, 2.2465507984161377]
        self.assertTrue(is_categorical(input_array))
        self.assertListEqual(input_array.order.values.tolist(), order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), dict(zip(order, [1] * 10)))
        self.assertListEqual(input_array.categories.tolist(), order)
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series(input_array).astype('category')))

    def test_111_create_categorical_with_too_many_categories(self):
        """Create a Categorical object which forces the NumberOfCategoriesWarning"""
        np.random.seed(987654321)
        self.assertWarnsCrossCompatible(NumberOfCategoriesWarning,
                                        lambda: Categorical(np.random.normal(size=51).astype('float32')))

    def test_112_create_categorical_from_integers(self):
        """Create a Categorical object from a numpy array of integers"""
        np.random.seed(987654321)
        input_array = Categorical(np.random.randint(-10, 10, 100))
        counts = [(-10, 4), (-9, 5), (-8, 8), (-7, 4), (-6, 7),
                  (-5, 3), (-4, 5), (-3, 6), (-2, 4), (-1, 2),
                  (0, 6), (1, 3), (2, 7), (3, 4), (4, 6),
                  (5, 6), (6, 5), (7, 3), (8, 7), (9, 5)]
        self.assertTrue(is_categorical(input_array))
        self.assertListEqual(input_array.order.values.tolist(), [i for i in range(-10, 10)])
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), dict(counts))
        self.assertListEqual(input_array.categories.tolist(), sorted(dict(counts).keys()))
        self.assertFalse(input_array.is_empty())

    # TODO: Add a unit test with a numeric list

    def test_113_create_large_categorical(self):
        """Create a Categorical object with a few categories but high counts"""
        input_array = Categorical([i for _ in range(0, 5000) for i in 'abc'], name='large')
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertEqual(input_array.name, 'large')
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 5000, 'b': 5000, 'c': 5000})
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c'])
        self.assertFalse(input_array.is_empty())

    def test_114_create_categorical_from_nested_lists(self):
        """Create a Categorical object from a nested list"""
        input_array = Categorical([['a', 'b', 'c'], ['d', 'e', 'f']])
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), dict(zip(['a', 'b', 'c', 'd', 'e', 'f'], [1] * 6)))
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd', 'e', 'f'])
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series(['a', 'b', 'c', 'd', 'e', 'f']).astype('category')))

    def test_115_create_categorical_from_2dim_array(self):
        """Make sure pandas throws an exception when trying to use a 2dim array"""
        input_array = np.array([['a', 'b', 'c'], ['d', 'e', 'f']])
        self.assertRaises(Exception, lambda: Categorical(input_array))

    def test_116_create_categorical_from_dict(self):
        """Create a Categorical object from a dictionary"""
        input_array = Categorical({'a': 1, 'b': 2, 'c': 3, 'd': 2})
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {1: 1, 2: 2, 3: 1})
        self.assertListEqual(input_array.categories.tolist(), [1, 2, 3])
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series({'a': 1, 'b': 2, 'c': 3, 'd': 2}).astype('category')))

    def test_117_create_categorical_with_multiindex(self):
        """Create a Categorical object with a multiindex"""
        index = MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'foo'), ('b', 'bar')])
        input_array = Categorical(Series([1, 2, 3, 4], index=index))
        self.assertTrue(is_categorical(input_array))
        self.assertListEqual(input_array.order.values.tolist(), [1, 2, 3, 4])
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {1: 1, 2: 1, 3: 1, 4: 1})
        self.assertListEqual(input_array.categories.tolist(), [1, 2, 3, 4])
        self.assertFalse(input_array.is_empty())

    def test_118_create_categorical_with_missing_data(self):
        """Create a Categorical object containing missing values"""
        ref = ["a", "b", "c", "d", np.nan, "b", "c", np.nan, "b"]
        input_array = Categorical(ref, name='test')
        self.assertTrue(is_categorical(input_array))
        self.assertIsNone(input_array.order)
        self.assertEqual(input_array.name, 'test')
        self.assertDictEqual(input_array.counts.to_dict(), {np.nan: 2, 'a': 1, 'b': 3, 'c': 2, 'd': 1})
        # TODO: Once NaN is displayed as a category, this test will need to be updated.
        self.assertListEqual(input_array.categories.tolist(), ['a', 'b', 'c', 'd'])
        self.assertFalse(input_array.is_empty())
        self.assertTrue(input_array.data.equals(Series(ref).astype('category')))

    def test_119_create_categorical_with_extra_order_categories(self):
        ref = ['a', 'b', 'c', 'b', 'a', 'd', 'c', 'c']
        order = ['e', 'd', 'c', 'b', 'a']
        input_array = Categorical(ref, order=order)
        self.assertTrue(is_categorical(input_array))
        self.assertListEqual(input_array.categories.tolist(), ['e', 'd', 'c', 'b', 'a'])
        self.assertListEqual(input_array.order, order)
        self.assertIsNone(input_array.name)
        self.assertDictEqual(input_array.counts.to_dict(), {'a': 2, 'b': 2, 'c': 3, 'd': 1, 'e': 0})

    def test_120_create_categorical_with_invalid_order_categories(self):
        ref = ['a', 'b', 'c', 'b', 'a', 'd', 'c', 'c']
        order = ['z', 'y', 'x', 'w']
        input_array = Categorical(ref, order=order)
        self.assertTrue(is_categorical(input_array))
        self.assertTrue(input_array.categories.tolist(), [])
        self.assertTrue(input_array.order, order)
        self.assertDictEqual(input_array.counts.to_dict(), dict([('z', 0), ('y', 0), ('x', 0), ('w', 0), (np.nan, 8)]))

if __name__ == '__main__':
    unittest.main()
