from functools import wraps
import inspect  # type: ignore
import types  # type: ignore
import re
from typing import Callable, List, Dict, TypeVar, Any
import toolz.curried as z
# from collections import OrderedDict

T = TypeVar('T')
START = 'START'
END = 'END'
enumx = lambda x: range(1, len(x) + 1)
enumxy = z.comp(enumerate, enumx)


def testprint(pred: bool) -> Callable[..., None]:
    f = print if pred else (lambda *x, **kw: None)  # type: ignore
    return f


def test_enumxy():
    x = [1, 2, 3, 4]
    y = [START, 1, 2, 3, 4, END]
    assert all([(x[i] == y[j]) for i, j in enumxy(x)])
    assert [y[i] for i in enumx(x)] == x


def const(x: T) -> Callable[..., T]:
    return lambda *a, **k: x


def todot(d: Dict[str, T]):
    d2 = lambda: None  # type: Any
    d2.__dict__ = d
    return d2


def numargs(f: Callable[..., Any]) -> int:
    argspec = inspect.getargspec(f)
    args, defs = argspec.args, (argspec.defaults or [])
    nargs, ndefs = len(args), len(defs)
    return nargs - ndefs


def justargs(f: Callable[..., Any]) -> List[str]:
    "Return arg names, ignoring those with defaults"
    return inspect.getargspec(f).args[:numargs(f)]


def test_numargs():
    def f1(a, b, c):
        pass

    def f2(a, b, c=None):
        pass

    assert justargs(f1) == ['a', 'b', 'c']
    assert justargs(f2) == ['a', 'b']
    assert numargs(f1) == 3
    assert numargs(f2) == 2


class FeatUtils(type):
    bookend = True

    @classmethod
    def sum2(cls, f: Callable[[str, str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
        "Return function that sums feature function `f` ∀ i in xbar"
        args_raw = inspect.getargspec(f).args

        @wraps(f)
        def fsum_book(xbar: List[str], ybar: List[str]) -> int:
            # if (len(ybar) - len(xbar) != 2) and isiter(xbar) and isiter(ybar):
            if cls.bookend:
                ybar_aug = cls.mkbookend(ybar)
            else:
                ybar_aug = ybar
            # p = ybar_aug == ['START', 'NPP', 'CD', 'DDD', 'END']
            # tt = testprint(p)
            # tt(f)
            # tt('ybar_aug:', ybar_aug)
            # tt('xbar:', xbar)
            # for xi, yi in enumxy(xbar):
            #     if not p:
            #         continue
            #     tt('yi', yi, 'xi', xi)
            #     tt('yp:', ybar_aug[yi - 1], 'y:', ybar_aug[yi], xbar, xi)
            #     tt('yp == START:', ybar_aug[yi - 1] == START, 'y == NNP:', ybar_aug[yi] == 'NNP', xbar, xi)
            #     tt('score: ', f(ybar_aug[yi - 1], ybar_aug[yi], xbar, xi))
                # print(ybar_aug[yi - 1], ybar_aug[yi], xi)

            if 'x' in args_raw:
                return sum(f(ybar_aug[yi - 1], ybar_aug[yi], xbar, xi) for xi, yi in enumxy(xbar))
            else:
                return sum(f(ybar_aug[yi - 1], ybar_aug[yi], None, xi) for xi, yi in enumxy(ybar))
        return fsum_book

    @classmethod
    def mk_sum(cls, f):
        a1 = ['yp', 'y', 'x', 'i']
        args = justargs(f)
        args_strip_score = [a.rstrip('_') for a in args]
        assert args_strip_score[:4] == a1, 'Function must have arguments {}. Not {}'.format(a1, args)
        f2 = cls.sum2(f)
        return f2

    @staticmethod
    def get_funcs(cls) -> Dict[str, Callable[..., bool]]:
        return {fname: f for fname, f in cls.__dict__.items()
                if not fname.startswith('_') and isinstance(f, types.FunctionType)}

    @staticmethod
    def mkbookend(ybar):
        return [START] + list(ybar) + [END]


def mk_word_tag(wd, tag):
    def f(yp_, y, x, i):
        return (x[i] == wd) and (y == tag)
    return f


class Fs():
    """Define feature functions here, with args `yp, y, x, i`
    To indicate an arg won't be used, append an underscore,
    e.g., `yp, y, x_, i` if the x argument is ignored
    """
    def post_mr(yp, y, x, i):  # optional keywords to not confuse mypy
        return (y == yp == 'NNP') and x[i - 1] == 'Mr.'
        # return (y == yp == 'NNP') and x[i - 1] == 'Mr.'

    def cap_nnp(yp_, y, x, i):
        return y == 'NNP' and x[i][0].isupper()

    def dig_cd(yp_, y, x, i, p=re.compile(r'[\d\.]+')):
        return y == 'CD' and bool(p.match(x[i]))

    def dt_in(yp, y, x_, i):
        return (yp == 'DT') and (y == 'IN')

    wd_to = mk_word_tag('to', 'TO')
    wd_of = mk_word_tag('of', 'IN')
    wd_for = mk_word_tag('for', 'IN')
    wd_in = mk_word_tag('in', 'IN')
    wd_a = mk_word_tag('a', 'DT')
    wd_the = mk_word_tag('the', 'DT')
    wd_and = mk_word_tag('and', 'CC')

    fst_dt = lambda yp_, y, x, i: (i == 0) and (y == 'DT')
    fst_nnp = lambda yp, y, x, i: (yp == START) and (y == 'NNP')
    # fst_nnp = lambda yp_, y, x, i: (i == 0) and (y == 'NNP')
    last_nn = lambda yp_, y, x, i: (i == len(x) - 1) and (y == 'NN')


fs = FeatUtils.get_funcs(Fs)
fsums = z.valmap(FeatUtils.mk_sum, fs)


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

# test_functions()


if __name__ == '__main__':
    test_functions()
    test_numargs()
