from os import path, getcwd
import unittest
import warnings

from .. data import is_iterable


class TestWarnings(unittest.TestCase):
    """A TestCase subclass with assertWarns substitute to cover python 2.7 which doesn't have an assertWarns method."""

    _seed = 987654321

    @property
    def save_path(self):
        if getcwd().split('/')[-1] == 'test':
            return './images/'
        elif getcwd().split('/')[-1] == 'sci_analysis':
            if path.exists('./setup.py'):
                return './sci_analysis/test/images/'
            else:
                return './test/images/'
        else:
            './'

    def assertWarnsCrossCompatible(self, expected_warning, *args, **kwargs):
        if 'message' in kwargs:
            _message = kwargs['message']
            kwargs.pop('message')
        else:
            _message = None
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            callable_obj = args[0]
            args = args[1:]
            callable_obj(*args, **kwargs)
            # This has to be done with for loops for py27 compatability
            for caught_warning in warning_list:
                self.assertTrue(issubclass(caught_warning.category, expected_warning))
            if _message is not None:
                if is_iterable(_message):
                    for i, m in enumerate(_message):
                        self.assertTrue(m in str(warning_list[i].message))
                else:
                    for caught_warning in warning_list:
                        self.assertTrue(_message in str(caught_warning.message))

    def assertNotWarnsCrossCompatible(self, expected_warning, *args, **kwargs):
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter('always')
            callable_obj = args[0]
            args = args[1:]
            callable_obj(*args, **kwargs)
            self.assertFalse(any(item.category == expected_warning for item in warning_list))
