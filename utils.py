import types
from functools import wraps
from typing import Callable, List
# from collections import OrderedDict
import inspect


class DecoMeta(type):
    def __new__(cls, name, bases, attrs):
        attrs['feats'] = feats = {}
        attrs['sum_feats'] = sum_feats = {}
        for attr_name, attr_value in attrs.items():
            if isinstance(attr_value, types.FunctionType):

                feats[attr_name] = f = attr_value
                newf = wraps(f)(staticmethod(cls.mk_sum(f)))
                sum_feats[attr_name] = attrs[attr_name] = newf

        return super(DecoMeta, cls).__new__(cls, name, bases, attrs)

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
        a1 = ['yi', 'x', 'i']
        a2 = ['yp', 'y', 'x', 'i']
        ags = inspect.getargspec(f)
        if ags.args == a1:
            f2 = cls.sum1(f)
        elif ags.args == a2:
            f2 = cls.sum2(f)
        else:
            raise TypeError('Function must have arguments {} or {}'.format(a1, a2))
        return f2


class F(metaclass=DecoMeta):
    def post_mr(yp, y, x=None, i=None):  # optional keywords to not confuse mypy
        return (y == yp == 'NNP') and x[i - 1] == 'Mr.'

    def cap_nnp(yi, x=None, i=None):
        return (yi == 'NNP') and x[i][0].isupper()

    def dt_in(yp, y, x=None, i=None):
        return (yp == 'DT') and (y == 'IN')

# F.post_mr()
F.dt_in('derp', ['DT', 'IN'])


def test_functions():
    x = 'However , Mr. Dillow said'.split()
    y1 = ['RB', ',']
    y2 = ['VBD']
    y = y1 + ['NNP', 'NNP'] + y2
    assert F.post_mr(x, y) == 1
    assert F.post_mr(x, y1 + ['NNPs', 'NNP'] + y2) == 0
    assert F.post_mr(x, y1 + ['NNP', 'NNPs'] + y2) == 0
    assert F.cap_nnp(x, y) == 2
    assert F.dt_in('derp', ['DT', 'IN']) == 1
    assert F.dt_in('derp', ['DT', 'INs']) == 0

test_functions()


if __name__ == '__main__':
    test_functions()
