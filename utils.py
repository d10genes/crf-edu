from itertools import repeat
from operator import methodcaller as mc
import toolz.curried as z
from typing import TypeVar, Dict, Callable  # , Tuple
from pandas import DataFrame
import pandas as pd
from py3k_imports import map

START = 'START'
END = 'END'
T = TypeVar('T')


def testprint(pred: bool) -> Callable[..., None]:
    f = print if pred else (lambda *x, **kw: None)  # type: ignore
    return f


class AttrDict(dict):
    "http://stackoverflow.com/a/14620633/386279"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def todot(d: Dict[str, T]):
    d2 = lambda: None  # type: Any
    d2.__dict__ = d
    return d2


def s2df(xs: List[Series]) -> DataFrame:
    return DataFrame({i: s for i, s in enumerate(xs)})


def debugu(ufunc, gmat, uadd, gf, pt, k):
    "Print some debug info for argmax calculation"
    ufunc.gmat = gmat
    ufunc.uadd = uadd
    pt('\n', k)
    pt(gf.xbar[k], )
    pt(gmat)
    pt('\nuadd')
    pt(uadd)


def eq(x):
    return lambda y: x == y


def sch(term, x=False, X=None, Y=None, Y_=None):
    "Search in X or Y for term, return matching input and output"
    f = eq(term) if isinstance(term, str) else term
    ss = X if x else Y
    for i, s in enumerate(ss):
        if any(f(t) for t in s):
            yield X[i], Y_[i]


# Show DataFrames and Series
def side_by_side(*ds, names=None):
    nms = iter(names) if names else repeat(None)
    dmultis = [side_by_side1(d, ctr=i, name=next(nms)) for i, d in enumerate(ds)]
    return pd.concat(dmultis, axis=1)


def side_by_side1(d, ctr=1, name=None):
    d = DataFrame(d.copy())
    d.columns = pd.MultiIndex.from_product([[name or ctr], list(d)])
    return d


def side_by_side_(*objs, **kwds):
    from pandas.core.common import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))


def ff(m):
    return side_by_side(m, m.idxmax(), m.max())


# Vector Data structures
class Derp(object):
    """Dummy object that keeps returning self/False to allow for flexible
    feature functions that can go out of bounds on x"""
    def __call__(self, *a, **kw):
        return self

    def __getattr__(self, *a, **kw):
        return self

    def __getitem__(self, *a, **kw):
        return self

    def __bool__(self):
        return False

    def __repr__(self):
        return 'OutOfBounds'

    def __add__(self, other):
        return other

    def __mul__(self, other):
        return 0

    def __radd__(self, other):
        return other

    def __rmul__(self, other):
        return 0

OutOfBounds = Derp()


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
        if isinstance(l, AugmentY):
            self._orig = l._orig
            self.aug = l.aug
            return
        assert isinstance(l, (list, tuple))
        self._orig = l
        self.aug = [START] + list(l) + [END]  # type: List

    def __repr__(self):
        return 'Aug' + repr(self._orig)


def memoize(f):
    """Memoization decorator for a function taking one or more arguments.
    http://code.activestate.com/recipes/578231-probably-the-fastest-memoization-decorator-in-the-/#c4
    """
    class memodict(dict):
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = f(*key)
            return ret

    return memodict().__getitem__


def process_corpus(corpus, sep='//'):
    psplit = lambda f: (lambda xs: map(z.comp(f, str.split), xs))
    linepairs = z.comp(z.map(mc('split', sep)), str.splitlines)(corpus)
    xs_, ys_ = zip(*linepairs)
    xs, ys = psplit(EasyList)(xs_), psplit(AugmentY)(ys_)
    return xs, ys
