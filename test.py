from utils import (EasyList, OutOfBounds, justargs, numargs, const, fs,
                   getmat, G, START, END, mk_word_tag, mkwts1)
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
    gft = G(fs=testfs, tags=stags, xbar=xt, ws=wst)
    resmat = getmat(gft(0), generic_names=True)

    assert all(resmat.NNP == 1)
    assert (resmat.drop('NNP', axis=1) == 0).all().all()
    return 0


def test_mats_2_args():
    xt = 'Mr. Derp has Three capped words'.split()
    testfs = z.keyfilter(lambda x: x in ('cap_nnp', 'post_mr'), fs)
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    gft = G(fs=testfs, tags=stags, xbar=xt, ws=wst)
    m0 = getmat(gft(0), generic_names=True)
    m1 = getmat(gft(1), generic_names=True)

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


def no_test_getu1(get_u, mlp):
    tgs = [START, 'TAG1', END]
    fs = {'eq_wd1': mk_word_tag('wd1', 'TAG1')}
    ytpred = [START, 'TAG1', END]
    x = EasyList(['wd1'])

    gf = G(fs=fs, tags=tgs, xbar=x, ws=mkwts1(fs))
    u, i = get_u(gf=gf, collect=True)
    assert (u.idxmax() == ytpred).all()
    assert u.iloc[:, -1].max() == 2


def no_test_getu2(get_u, mlp):
    tgs = [START, 'TAG1', END]
    x2 = EasyList(['wd1', 'pre-end'])
    fs = {'eq_wd1': mk_word_tag('wd1', 'TAG1'),
          'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END)}
    ws = z.merge(mkwts1(fs), {'pre_endx': 3})
    gf2 = G(fs=fs, tags=tgs, xbar=x2, ws=ws)
    assert all(getmat(gf2(3))[END] == 3)
    no_test_getu2.gf2 = gf2
    no_test_getu2.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)
    # print(u2)
    assert (u2.idxmax() == [START, 'TAG1', 'TAG1', END]).all()
    assert u2.iloc[:, -1].max() == 5
    assert mlp(i2) == ['START', 'TAG1', 'TAG1', 'END']
    return u2, i2


def no_test_getu3(get_u, mlp):
    tgs = [START, 'TAG1', 'PENULTAG', END]
    fs = {
        'eq_wd1': mk_word_tag('wd1', 'TAG1'),
        # 'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END),
        'pre_endy': lambda yp, y, x, i: (yp == 'PENULTAG') and (y == END),
        'start_nonzero': lambda yp, y, x, i: (y == START) and (i != 0),
        'start_zero': lambda yp, y, x, i: (y == START) and (i == 0),
        'end_nonend': lambda yp, y, x, i: (y == END) and (i != (len(x) + 1)),
        'end_end': lambda yp, y, x, i: (y == END) and (i == (len(x) + 1)),
    }
    ws = z.merge(mkwts1(fs), {'pre_endy': 3, 'start_nonzero': -1, 'end_nonend': -1})
    x2 = EasyList(['wd1', 'pre-end', 'whatevs'])
    gf2 = G(fs=fs, tags=tgs, xbar=x2, ws=ws)
    no_test_getu3.gf2 = gf2
    no_test_getu3.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)
    assert mlp(i2) == ['START', 'TAG1', 'PENULTAG', 'PENULTAG', 'END']
    return u2, i2
