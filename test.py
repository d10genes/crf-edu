from crf import (justargs, numargs, const, fs, fsums,
                 getmat, G, mk_word_tag, mkwts1)
from utils import OutOfBounds, EasyList, START, END, todot
import toolz.curried as z
from pandas.util.testing import assert_frame_equal


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


def test_mats_replace():
    xt = 'Hi this has Two capped words'.split()
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    testfs = dict(cap_nnp=fs['cap_nnp'])
    gft = G(fs=testfs, tags=stags, xbar=xt, ws=wst)
    resmat = gft(1).mat
    assert all(resmat.NNP == 1)
    assert (resmat.drop('NNP', axis=1) == 0).all().all()
    gft2 = gft._replace(ws={'cap_nnp': 2})
    assert all(gft2(1).mat.NNP == 2)

    return 0


def test_mats_2_args():
    xt = 'Mr. Derp has Three capped words'.split()
    testfs = z.keyfilter(lambda x: x in ('cap_nnp', 'post_mr'), fs)
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    gft = G(fs=testfs, tags=stags, xbar=xt, ws=wst)
    m0 = getmat(gft(1), generic_names=True)
    m1 = getmat(gft(2), generic_names=True)

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


# Test argmax_y
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
    assert all(gf2(3).mat[END] == 3)
    no_test_getu2.gf2 = gf2
    no_test_getu2.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)

    assert (u2.idxmax() == [START, 'TAG1', 'TAG1', END]).all()
    assert u2.iloc[:, -1].max() == 5
    assert mlp(i2) == ['START', 'TAG1', 'TAG1', 'END']
    return u2, i2


def mk_fst():
    tgs = [START, 'DUMMY', 'TAG1', 'TAG2', 'PENULTAG', END]
    fs = {
        'eq_wd1': mk_word_tag('wd1', 'TAG1'),
        # 'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END),
        'pre_endy': lambda yp, y, x, i: (yp == 'PENULTAG') and (y == END),
        'start_nonzero': lambda yp, y, x, i: (y == START) and (i != 0),
        'start_zero': lambda yp, y, x, i: (y == START) and (i == 0),
        'end_nonend': lambda yp, y, x, i: (y == END) and (i != (len(x) + 1)),
        'end_end': lambda yp, y, x, i: (y == END) and (i == (len(x) + 1)),
        't1_t2': lambda yp, y, x, i: (yp == 'TAG1') and (y == 'TAG2'),
    }
    x2 = EasyList(['wd1', 'pre-end', 'whatevs'])
    return fs, tgs, x2


def no_test_getu3(get_u, mlp):
    fs, tgs, x2 = mk_fst()
    ws = z.merge(mkwts1(fs), {'pre_endy': 3, 'start_nonzero': -1, 'end_nonend': -1})
    gf2 = G(fs=fs, tags=tgs, xbar=x2, ws=ws)
    no_test_getu3.gf2 = gf2
    no_test_getu3.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)

    assert mlp(i2) == ['START', 'TAG1', 'TAG2', 'PENULTAG', 'END']
    return u2, i2


def test_functions():
    f = todot(fsums)
    split_args_ = lambda *x: zip(*map(str.split, x))  # ['a 1', 'b 2'] -> [('a', 'b'), ('1', '2')]
    split_args = lambda f: (lambda *a: f(*split_args_(*a)))  # decorate f by preprocessing args with split_args_
    g = todot(z.valmap(split_args, fsums))
    x = 'However , Mr. Dillow said'.split()
    y = ['RB', ',', 'NNP', 'NNP', 'VBD']
    y1 = ['RB', ',']
    y2 = ['VBD']
    # y = y1 + ['NNP', 'NNP'] + y2
    assert g.post_mr('However RB', ', ,', 'Mr. NNP', 'Dillow NNP', 'said VBD') == 1
    assert f.post_mr(*split_args_('However RB', ', ,', 'Mr. NNP', 'Dillow NNP', 'said VBD')) == 1
    assert f.post_mr(x, y) == 1
    assert f.post_mr(x, y1 + ['NNPs', 'NNP'] + y2) == 0
    assert f.post_mr(x, y1 + ['NNP', 'NNPs'] + y2) == 0
    assert f.cap_nnp(x, y) == 2
    assert f.cap_nnp([' ', 'Mr.', 'low', 'Up'],
                     [' ', 'NNP', 'NNP', ' ']) == 1
    assert f.dt_in('derp', ['DT', 'IN']) == 1
    assert f.dt_in('derp', ['DT', 'INs']) == 0
    assert f.dig_cd(['123', 'hi', 'the'], ['CD', 'INs', 'TAG']) == 1
    assert f.dig_cd(['123', '1.23', 'the'], ['CD', 'CD', 'TAG']) == 2
    assert f.wd_to(['to', '1.23', 'the'], ['TO', 'CD', 'TAG']) == 1
    assert f.last_nn(['to', '1.23', 'the'], ['TO', 'CD', 'TAG']) == 0
    assert f.last_nn(['to', '1.23', 'the'], ['TO', 'CD', 'NN']) == 1
    assert f.fst_dt(['to', '1.23', 'the'], ['DT', 'CD', 'NN']) == 1
    assert f.fst_dt(['to', '1.23', 'the'], ['DT2', 'CD', 'NN']) == 0
    assert f.fst_nnp(['to', '1.23', 'ddd'], ['NNP', 'CD', 'DDD']) == 1
