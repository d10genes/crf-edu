import types  # type: ignore
from functools import wraps
from typing import Callable, List, Dict
import inspect  # type: ignore
import toolz.curried as z
# from collections import OrderedDict


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
        return fsum

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

    def cap_nnp(yp, y, x, i):
        return ((yp == 'NNP') or y == 'NNP') and x[i][0].isupper()

    def dt_in(yp, y, x_, i):
        return (yp == 'DT') and (y == 'IN')


fs = FeatUtils.get_funcs(Fs)
fsums = z.valmap(FeatUtils.mk_sum, fs)


def test_functions():
    x = 'However , Mr. Dillow said'.split()
    y1 = ['RB', ',']
    y2 = ['VBD']
    y = y1 + ['NNP', 'NNP'] + y2
    assert fsums['post_mr'](x, y) == 1
    assert fsums['post_mr'](x, y1 + ['NNPs', 'NNP'] + y2) == 0
    assert fsums['post_mr'](x, y1 + ['NNP', 'NNPs'] + y2) == 0
    assert fsums['cap_nnp'](x, y) == 2
    assert fsums['dt_in']('derp', ['DT', 'IN']) == 1
    assert fsums['dt_in']('derp', ['DT', 'INs']) == 0

test_functions()


if __name__ == '__main__':
    test_functions()
