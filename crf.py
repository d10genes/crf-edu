from functools import wraps
import inspect  # type: ignore
import types  # type: ignore
import re
from typing import Callable, List, Dict, TypeVar, Any  # , Tuple
import toolz.curried as z
import numpy.random as nr
from pandas import DataFrame  # type: ignore
from collections import namedtuple, OrderedDict
from utils import EasyList, AugmentY, START, END

T = TypeVar('T')

mkwts1 = lambda dct, const=1: {k: const for k in dct}
funcdat = namedtuple('funcdat', 'fs tags xbar ws i')
funcdat.__new__.__defaults__ = (None,)


def rand_weights(dct, seed=0):
    nr.seed(seed)
    return dict(zip(sorted(dct), nr.randn(len(dct))))


class FuncSums(object):
    def __call__(self, y: str, x: str) -> float:  # #type: Callable[[str, str], float]
        pass
    tags = ['']
    i = 1


def const(x: T) -> Callable[..., T]:
    return lambda *a, **k: x


def numargs(f: Callable[..., Any]) -> int:
    params = inspect.signature(f).parameters.values()
    return len([p for p in params if p.default == inspect._empty])
    # argspec = inspect.getargspec(f)
    # args, defs = argspec.args, (argspec.defaults or [])
    # nargs, ndefs = len(args), len(defs)
    # return nargs - ndefs


def justargs(f: Callable[..., Any]) -> List[str]:
    "Return arg names, ignoring those with defaults"
    return list(inspect.signature(f).parameters.keys())[:numargs(f)]
    # return inspect.getargspec(f).args[:numargs(f)]


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


class G(object):
    """
    fs=None, tags=None, xbar: List=None, ws=None
    For dict of functions, corresponding weights, tags and xbar,
    this class holds functions for generating g_i matrix, which sums all of the
    functions times weights for a given i with inputs
    """
    def __init__(self, fs=None, tags=None, xbar: List=None, ws=None, fdat=None):
        assert (fs and tags and xbar) or fdat, 'Either fdat or all the rest required'
        if not fdat:
            xbar_ = xbar if isinstance(xbar, EasyList) else EasyList(xbar)
            fdat = funcdat(fs=fs, tags=sorted(tags), xbar=xbar_, ws=ws or mkwts1(fs))
        self._data = fdat

    def __repr__(self):
        return 'G' + repr(self._data)[7:]

    def _replace(self, **kwds):
        res = self._data._replace(**kwds)
        return G(fdat=res)

    def __getattr__(self, attr):
        return getattr(self._data, attr)

    class Gi(funcdat):
        def __call__(self, yp, y):
            return sum([f(yp, y, self.xbar, self.i) * self.ws[fn] for fn, f in self.fs.items()])

        @property
        def mat(self):
            return getmat(self)

    def __call__(self, i):
        return self.Gi(*self._data._replace(i=i))

    def both(self, g2):
        d1, d2 = self._data.__getattribute__, g2._data.__getattribute__
        d = OrderedDict([(f, (d1(f), d2(f))) for f in self._data._fields])
        return d

    def diff(self, g2):
        bth = self.both(g2)
        return z.valfilter(lambda tup: tup[0] != tup[1], bth)

    def __dir__(self):
        return super().__dir__() + list(self._data._fields)
        # return dir(self) + list(self._data.fields)


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
    def mk_sum(cls, f: Callable[[str, str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
        def fsum_book(xbar: List[str], ybar: List[str]) -> int:
            """Convert function of f(yp, y, xbar, i) to one that sums over all
            i's: F(xbar, ybar)
            """
            if cls.bookend:
                yb = AugmentY(ybar)
            else:
                yb = ybar
            xb = EasyList(xbar)
            return sum(f(yb.aug[i - 1], yb.aug[i], xb, i) for i in range(1, len(yb.aug)))
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
        return y == 'CD' and x[i] and bool(p.match(x[i]))

    def dt_in(yp: 'DT', y: 'IN', x_, i):
        return (yp == 'DT') and (y == 'IN')

    wd_to = mk_word_tag('to', 'TO')
    wd_of = mk_word_tag('of', 'IN')
    wd_for = mk_word_tag('for', 'IN')
    wd_in = mk_word_tag('in', 'IN')
    wd_a = mk_word_tag('a', 'DT')
    wd_the = mk_word_tag('the', 'DT')
    wd_and = mk_word_tag('and', 'CC')

    fst_dt = lambda yp, y, x_, i: (yp == START) and (y == 'DT')
    fst_nnp = lambda yp, y, x_, i: (yp == START) and (y == 'NNP')
    last_nn = lambda yp, y, x_, i: (yp == 'NN') and (y == END)


fs = FeatUtils.get_funcs(Fs)
fsums = z.valmap(FeatUtils.mk_sum, fs)

# if __name__ == '__main__':
#     test_functions()

# $$M_t (u, v) = \exp g_t(u, v) \in ℝ^m$$
# where $M_1$ only defined for $u=START$,
# where $M_{n+1}$ only defined for $v=END$
#     $$M_{12} = M_1 M_2$$ (matrix multiplication)
# i.e.,
#     $$M_{12}(START, w) = \sum_v M_1(START, v)M_2(v,w)$$
#     $$= \sum_{y_1} \exp \left[g_1(START, y_1) +  g_2(y_1, w) \right]$$
#     $$M_{123}(START, w) = \sum_{y_2} M_{12}(START, {y_2})M_3({y_2},w)$$
