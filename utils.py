import types  # type: ignore
from functools import wraps
from typing import Callable, List, Dict, TypeVar, Any
import inspect  # type: ignore
import toolz.curried as z
# from collections import OrderedDict

T = TypeVar('T')


def const(x: T) -> Callable[..., T]:
    return lambda *a, **k: x


def todot(d: Dict[str, T]):
    d2 = lambda: None  # type: Any
    d2.__dict__ = d
    return d2


class FeatUtils(type):
    @staticmethod
    def sum1(f: Callable[[str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
        "Return function that sums feature function `f` ∀ i in xbar"
        @wraps(f)
        def fsum(xbar: List[str], ybar: List[str]) -> int:
            return sum(f(yi, xbar, i) for i, yi in enumerate(ybar))
        return fsum

    @staticmethod
    def sum2(f: Callable[[str, str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
        "Return function that sums feature function `f` ∀ i in xbar"
        @wraps(f)
        def fsum(xbar: List[str], ybar: List[str]) -> int:
            return sum(f(ybar[i - 1], ybar[i], xbar, i) for i, yi in enumerate(ybar[:-1], 1))

        @wraps(f)
        def fsum1(xbar: List[str], ybar: List[str]) -> int:
            return sum(f(None, yi, xbar, i) for i, yi in enumerate(ybar))
        ignore_yp = 'yp_' in inspect.getargspec(f).args

        return fsum1 if ignore_yp else fsum

    @classmethod
    def mk_sum(cls, f):
        a1 = ['yp', 'y', 'x', 'i']

        args = inspect.getargspec(f).args
        args_strip_score = [a.rstrip('_') for a in args]
        assert args_strip_score == a1, 'Function must have arguments {}. Not {}'.format(a1, args)
        f2 = cls.sum2(f)
        return f2

    @staticmethod
    def get_funcs(cls) -> Dict[str, Callable[..., bool]]:
        return {fname: f for fname, f in cls.__dict__.items()
                if not fname.startswith('_') and isinstance(f, types.FunctionType)}


class Fs():
    """Define feature functions here, with args `yp, y, x, i`
    To indicate an arg won't be used, append an underscore,
    e.g., `yp, y, x_, i` if the x argument is ignored
    """
    def post_mr(yp, y, x, i):  # optional keywords to not confuse mypy
        return (y == yp == 'NNP') and x[i - 1] == 'Mr.'

    def cap_nnp(yp_, y, x, i):
        # return ((yp == 'NNP') or (y == 'NNP')) and x[i][0].isupper()
        return y == 'NNP' and x[i][0].isupper()

    def dt_in(yp, y, x_, i):
        return (yp == 'DT') and (y == 'IN')


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

test_functions()


if __name__ == '__main__':
    test_functions()
