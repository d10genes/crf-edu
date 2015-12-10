# this makes flake8 checker choke
from pandas import Series, DataFrame
from pandas.util.testing import assert_frame_equal


def test_matmul():
    s1, s2 = Series([1, 2, 3]), Series([1, 2, 3])
    assert (s1 @ s2) == 14

    s = Series([1, 2])
    d = DataFrame([[1, 1], [2, 2]])
    assert all(s @ d == [5, 5])
    assert_frame_equal(d @ d, DataFrame([[3, 3], [6, 6]]))
    assert all(d @ s == [3, 6])
