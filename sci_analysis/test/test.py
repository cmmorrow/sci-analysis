import numpy.ma as ma
import pandas as pd

from ..analysis.analysis import *
from ..operations.data_operations import *

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

passing = 0

for name, test in inputs.iteritems():
    try:
        assert is_array(test)
        #assert is_group(test) or is_dict_group(test)
        print "PASS: " + name
        passing += 1
    except (AssertionError, TypeError):
        print "FAIL: " + name


print "----------------------------"
print "Passing tests: " + str(passing) + " of " + str(len(inputs))

# Vector testing
print ""
test_list = [1.0, "2", '3.0', "four", 5]
try:
    assert Vector(test_list, "test")
    print "PASS Vector test"
    print Vector(test_list, "test").data
except AssertionError:
    print "FAIL Vector test"

# Drop nan testing
print ""
try:
    v = Vector(test_list, name="test")
    assert any(drop_nan(v))
    print "PASS Drop nan test"
    print drop_nan(Vector(test_list))
except AssertionError:
    print "FAIL Drop nan test"


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