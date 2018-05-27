from os import path, getcwd
import unittest
from warnings import catch_warnings, simplefilter

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
        with catch_warnings(record=True) as warning_list:
            simplefilter('always')
            callable_obj = args[0]
            args = args[1:]
            callable_obj(*args, **kwargs)
            print('org message: {}'.format(_message))
            print('expected warning: {}'.format(expected_warning))
            print('warning list: {}'.format(warning_list))
            print('warning category: {}'.format(warning_list[0].category))
            self.assertTrue(any(item.category == expected_warning for item in warning_list))
            if _message is not None:
                if is_iterable(_message):
                    for i, m in enumerate(_message):
                        self.assertTrue(m in str(warning_list[i].message))
                else:
                    self.assertTrue(any(_message in str(item.message) for item in warning_list))

    def assertNotWarnsCrossCompatible(self, expected_warning, *args, **kwargs):
        with catch_warnings(record=True) as warning_list:
            simplefilter('always')
            callable_obj = args[0]
            args = args[1:]
            callable_obj(*args, **kwargs)
            self.assertFalse(any(item.category == expected_warning for item in warning_list))
