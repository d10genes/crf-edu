import builtins
from functools import wraps, reduce
from importlib import reload


def listify(f):
    @wraps(f)
    def wrapper(*a, **k):
        return list(f(*a, **k))
    return wrapper


for fn in 'map range filter zip'.split():
    exec('i{f} = builtins.{f}'.format(f=fn))
    exec('{f} = listify(builtins.{f})'.format(f=fn))
