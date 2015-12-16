from bitarray import bitarray
from voluptuous import Schema


class Tracer(object):
    def __init__(self):
        self.eq = []

    def __eq__(self, other):
        self.eq.append(other)
        return True


def dependency(f, toint=True):
    """Return booleans for which args the function is dependent on, based on the parameter names.
    If the first arg name ends with an underscore, the first return value will false.
    -> (bool_yprev, bool_y, bool_xi)
    """
    res = tuple([not p.endswith('_') for p in inspect.signature(f).parameters][:3])
    if toint:
        return int(bitarray(res).to01(), 2)
    return res


class Flag(int):
    def __new__(cls, name: str, i):
        return super().__new__(cls, i)

    def __init__(self, name, i):
        self.name = name
        self.i = i
        Flag.flags[i] = self
        super().__init__()

    def __repr__(self):
        return '{}:{:b}'.format(self.name, self.i)

    def isin(self, x):
        return x & self == self

    flags = {}


Y1 = Flag('Y1', 2 ** 2)
Y2 = Flag('Y2', 2 ** 1)
XI = Flag('XI', 2 ** 0)
Y12 = Flag('Y12', Y1 | Y2)
Ynone = Flag('Ynone', 0)


dd = defaultdict(lambda: defaultdict(list))
dd[Ynone] = []


def categorizey(f, ys=dd):  # tup: (bool, bool, bool),
#     y1, y2, x = dependency(f)
    dep = dependency(f)
    yparg, yarg = Tracer(), Tracer()

    f(yparg, yarg, None, None)

    if Y12.isin(dep):
        [y1val], [y2val] = yparg.eq, yarg.eq
        ys[Y12][(y1val, y2val)].append(f)
    elif Y1.isin(dep):
        [y1val] = yparg.eq
        ys[Y1][y1val].append(f)
    elif Y2.isin(dep):
        [y2val] = yarg.eq
        ys[Y2][y2val].append(f)
    else:
        ys[Ynone].append(f)
    return ys

    return yparg, yarg, y1val

# yparg, yarg =
categorizey(f)


xs = {}
def categorizefunc(f, tags=None, xs=xs):
    dep = dependency(f)
    if not XI.isin(dep):
        xs['nox'] = categorizey(f, xs['nox'])