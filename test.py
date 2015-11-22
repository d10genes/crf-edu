from utils import (EasyList, OutOfBounds, justargs, numargs, const, fs, mkgf,
                   getmat)
import toolz.curried as z
from pandas.util.testing import assert_frame_equal


# enumx = lambda x: range(1, len(x) + 1)
# enumxy = z.comp(enumerate, enumx)
#
#
# def test_enumxy():
#     x = [1, 2, 3, 4]
#     y = [START, 1, 2, 3, 4, END]
#     assert all([(x[i] == y[j]) for i, j in enumxy(x)])
#     assert [y[i] for i in enumx(x)] == x


def test_easylist():
    el = EasyList([1, 2, 3])
    assert el[0] == OutOfBounds
    assert el[1] == 1
    assert el[2] == 2
    assert el[3] == 3
    assert el[4] == OutOfBounds
    assert el[-1] == 3
    assert el[-2] == 2


def test_numargs():
    def f1(a, b, c):
        pass

    def f2(a, b, c=None):
        pass

    assert justargs(f1) == ['a', 'b', 'c']
    assert justargs(f2) == ['a', 'b']
    assert numargs(f1) == 3
    assert numargs(f2) == 2


def test_mats():
    xt = 'Hi this has Two capped words'.split()
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    testfs = dict(cap_nnp=fs['cap_nnp'])
    gft = mkgf(wst, testfs, stags, xt)
    resmat = getmat(gft(0))

    assert all(resmat.NNP == 1)
    assert (resmat.drop('NNP', axis=1) == 0).all().all()
    return 0


def test_mats_2_args():
    xt = 'Mr. Derp has Three capped words'.split()
    testfs = z.keyfilter(lambda x: x in ('cap_nnp', 'post_mr'), fs)
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    gft = mkgf(wst, testfs, stags, xt)
    m0 = getmat(gft(0))
    m1 = getmat(gft(1))

    # First position should be the same
    assert all(m0.NNP == 1)
    assert (m0.drop('NNP', axis=1) == 0).all().all()

    # Second should get additional point from Mr. feature in position
    # y-1 == NNP, y == NNP
    assert m1.NNP.NNP == 2
    # Subtracting that should give same matrix as original
    m1c = m1.copy()
    m1c.loc['NNP', 'NNP'] = m1.NNP.NNP - 1
    assert_frame_equal(m1c, m0)
    return gft
