from functools import wraps
from typing import Callable, List
# from collections import OrderedDict
import inspect


def sum1(f: Callable[[str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
    "Return function that sums feature function `f` ∀ i in xbar"
    @wraps(f)
    def fsum(xbar: List[str], ybar: List[str]) -> int:
        return sum(f(yi, xbar, i) for i, yi in enumerate(ybar))
    return fsum


def sum2(f: Callable[[str, str, List[str], int], int]) -> Callable[[List[str], List[str]], int]:
    "Return function that sums feature function `f` ∀ i in xbar"
    @wraps(f)
    def fsum(xbar: List[str], ybar: List[str]) -> int:
        return sum(f(ybar[i - 1], ybar[i], xbar, i) for i, yi in enumerate(ybar[:-1], 1))
    return fsum


def mk_sum(f):
    a1 = ['yi', 'x', 'i']
    a2 = ['yp', 'y', 'x', 'i']
    ags = inspect.getargspec(f)
    if ags.args == a1:
        f2 = sum1(f)
    elif ags.args == a2:
        f2 = sum2(f)
    else:
        raise TypeError('Function must have arguments {} or {}'.format(a1, a2))
    mk_sum.fs.append(f2)
    return f2


mk_sum.fs = []


@mk_sum
def post_mr(yp, y, x, i):
    return (y == yp == 'NNP') and x[i - 1] == 'Mr.'


@mk_sum
def cap_nnp(yi, x, i):
    return (yi == 'NNP') and x[i][0].isupper()


@mk_sum
def dt_in(yp, y, x, i):
    return (yp == 'DT') and (y == 'IN')


def test_functions():
    x = 'However , Mr. Dillow said'.split()
    y1 = ['RB', ',']
    y2 = ['VBD']
    y = y1 + ['NNP', 'NNP'] + y2
    assert post_mr(x, y) == 1
    assert post_mr(x, y1 + ['NNPs', 'NNP'] + y2) == 0
    assert post_mr(x, y1 + ['NNP', 'NNPs'] + y2) == 0
    assert cap_nnp(x, y) == 2
    assert dt_in('derp', ['DT', 'IN']) == 1
    assert dt_in('derp', ['DT', 'INs']) == 0


if __name__ == '__main__':
    test_functions()
