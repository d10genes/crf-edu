from functools import wraps
import inspect  # type: ignore
import types  # type: ignore
import re
from typing import Callable, List, Dict, TypeVar, Any  # , Tuple
import toolz.curried as z
from pandas import DataFrame  # type: ignore
from collections import namedtuple
# from collections import OrderedDict

T = TypeVar('T')
START = 'START'
END = 'END'
mkwts1 = lambda dct: {k: 1 for k in dct}
OutOfBounds = object()
funcdat = namedtuple('funcdat', 'fs tags xbar ws i')
funcdat.__new__.__defaults__ = (None,)


class EasyList(list):
    """1-based indexed list that is forgiving with trying to access out of bounds.
    This way generic functions can be used that try to access arbitrary relative positions."""
    def __init__(self, lst=[], verbose=False):
        self.verbose = verbose
        super().__init__(lst)

    def __getitem__(self, ix):
        if ix < 0:
            ix_ = ix
        elif not ix:
            ix_ = len(self)
        else:
            ix_ = ix - 1
        try:
            return super().__getitem__(ix_)
        except IndexError:
            if self.verbose:
                print('{} out of bounds with list of length {}'.format(ix, len(self)))
            return OutOfBounds

    def __repr__(self):
        res = super().__repr__()
        return 'EasyList' + res

    def __setitem__(self, ix, val):
        raise NotImplementedError


class AugmentY(object):
    def __init__(self, l):
        self._orig = l._orig if isinstance(l, AugmentY) else l
        self.aug = [START] + list(l) + [END]  # type: List

    def __repr__(self):
        return 'Aug' + repr(self._orig)


class FuncSums(object):
    def __call__(self, y: str, x: str) -> float:  # #type: Callable[[str, str], float]
        pass
    tags = ['']
    i = 1


def testprint(pred: bool) -> Callable[..., None]:
    f = print if pred else (lambda *x, **kw: None)  # type: ignore
    return f


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


def debugger(f):
    @wraps(f)
    def debug(*a, **kw):
        try:
            return f(*a, **kw)
        except Exception as e:
            print(f, 'args:', a, 'kwargs:', kw)
            print('attrs:', debug.base_f.__dict__)
            raise(e)

    debug.base_f = f
    return debug


class G(funcdat):
    """For dict of functions, corresponding weights, tags and xbar,
    this class holds functions for generating g_i matrix, which sums all of the
    functions times weights for a given i with inputs
    """
    def __new__(cls, fs=None, tags=None, xbar: List=None, ws=None):
        # xbar_ = xbar if isinstance(xbar, EasyList) else EasyList(xbar)
        # self.data = funcdat(fs=fs, tags=sorted(tags), xbar=xbar_, ws=ws or mkwts1(fs))
        xbar_ = xbar if isinstance(xbar, EasyList) else EasyList(xbar)
        self = super().__new__(cls, fs, sorted(tags), xbar_, ws)
        return self

    def __init__(self, *args, **kwargs):
        # xbar_ = xbar if isinstance(xbar, EasyList) else EasyList(xbar)
        # self.data = funcdat(fs=fs, tags=sorted(tags), xbar=xbar_, ws=ws or mkwts1(fs))
        self.data = funcdat(*args, **kwargs)

    class Gi(funcdat):
        def __call__(self, yp, y):
            return sum(f(yp, y, self.xbar, self.i) * self.ws[fn] for fn, f in self.fs.items())

        @property
        def mat(self):
            return getmat(self)

    def __call__(self, i):
        return self.Gi(*self.data._replace(i=i))


def mkgf(ws, fs, tags, xbar):
    tags = sorted(tags)

    def gf(i):
        # @debugger
        def gfi(yp, y):
            return sum(f(yp, y, xbar, i) * ws[fn] for fn, f in fs.items())
        # print(gfi.base_f.__dict__)
        gfi.tags = tags  # gfi.base_f.tags =
        gfi.i = i  # gfi.base_f.i =
        return gfi
    gf.xbar = xbar
    gf.tags = tags
    return gf


def getmat(gf: FuncSums, generic_names=False) -> Any:
    "((yp, y) -> float) -> (Df[Yprev x Y] -> float)"
    tags = gf.tags
    i = gf.i
    df = DataFrame({ytag: {ytag_prev: gf(ytag_prev, ytag) for ytag_prev in gf.tags}
                    for ytag in gf.tags})  # type: ignore
    if generic_names:
        df.columns.name, df.index.name = 'Y', 'Yprev'
    else:
        df.columns.name, df.index.name = 'y{}'.format(i), 'y{}'.format(i - 1)

    return df[tags].ix[tags]


class FeatUtils(type):
    bookend = True

    @classmethod
    def mk_sum2(cls, f: Callable[[str, str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
        def fsum_book(xbar: List[str], ybar: List[str]) -> int:
            """Convert function of f(yp, y, xbar, i) to one that sums over all
            i's: F(xbar, ybar)
            """
            yb = AugmentY(ybar)
            xb = EasyList(xbar)
            # print(xb)
            # for i in range(1, len(yb.aug)):
            #     print(i, end=' ')
            #     print('yp:', yb.aug[i - 1], 'y:', yb.aug[i], 'x[i]:', xb[i])
            return sum(f(yb.aug[i - 1], yb.aug[i], xb, i) for i in range(1, len(yb.aug)))
            fsum_book.base_f = f
            fsum_book.__doc__ = '\n'.join([f.__doc__, fsum_book.__doc__])
        return fsum_book

    @classmethod
    def mk_sum(cls, f: Callable[[str, str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
        def fsum_book(xbar: List[str], ybar: List[str]) -> int:
            """Convert function of f(yp, y, xbar, i) to one that sums over all
            i's: F(xbar, ybar)
            """
            yb = AugmentY(ybar)
            xb = EasyList(xbar)
    #         print(xb)
    #         for i in range(1, len(yb.aug)):
    #             print(i, end=' ')
    #             print('yp:', yb.aug[i - 1], 'y:', yb.aug[i], 'x[i]:', xb[i])
            return sum(f(yb.aug[i - 1], yb.aug[i], xb, i) for i in range(1, len(yb.aug)))
            fsum_book.base_f = f
            fsum_book.__doc__ = '\n'.join([f.__doc__, fsum_book.__doc__])
        return fsum_book

    @staticmethod
    def check_args(f):
        a1 = ['yp', 'y', 'x', 'i']
        args = justargs(f)
        args_strip_score = [a.rstrip('_') for a in args]
        assert args_strip_score[:4] == a1, 'Function must have arguments {}. Not {}'.format(a1, args)

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
    f.__name__ = '{}_eq_{}'.format(wd, tag)
    f.__doc__ = '(x[i] == {}) and (y == {})'.format(wd, tag)
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

    fst_dt = lambda yp, y, x, i: (yp == START) and (y == 'DT')
    fst_nnp = lambda yp, y, x, i: (yp == START) and (y == 'NNP')
    last_nn = lambda yp, y, x, i: (yp == 'NN') and (y == END)


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


class AttrDict(dict):
    "http://stackoverflow.com/a/14620633/386279"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


if __name__ == '__main__':
    test_functions()


# $$M_t (u, v) = \exp g_t(u, v) \in ℝ^m$$
# where $M_1$ only defined for $u=START$,
# where $M_{n+1}$ only defined for $v=END$
#     $$M_{12} = M_1 M_2$$ (matrix multiplication)
# i.e.,
#     $$M_{12}(START, w) = \sum_v M_1(START, v)M_2(v,w)$$
#     $$= \sum_{y_1} \exp \left[g_1(START, y_1) +  g_2(y_1, w) \right]$$
#     $$M_{123}(START, w) = \sum_{y_2} M_{12}(START, {y_2})M_3({y_2},w)$$
