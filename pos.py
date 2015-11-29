
# coding: utf-8

# In[ ]:

from py3k_imports import * 
import project_imports3; reload(project_imports3); from project_imports3 import *

import warnings
warnings.filterwarnings('ignore')

pu.psettings(pd)
pd.options.display.width = 150   # 200
get_ipython().magic('matplotlib inline')


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from collections import defaultdict, Counter
import inspect
from typing import List, Dict
Df = Dict
Y = str

if sys.version_info.major > 2:
    unicode = str


# In[ ]:

Series.__matmul__ = Series.dot
DataFrame.__matmul__ = DataFrame.dot


# In[ ]:

def test_matmul():
    s1, s2 = Series([1, 2, 3]), Series([1, 2, 3])
    assert (s1 @ s2) == 14
        
    s = Series([1, 2])
    d = DataFrame([[1, 1], [2, 2]])
    assert all(s @ d == [5, 5])
    assert_frame_equal(d @ d, DataFrame([[3, 3], [6, 6]]))
    assert all(d @ s == [3, 6])
test_matmul()


# from py3k_imports import *
# from project_imports3 import *
# import pandas as pd
# import seaborn as sns
# import autograd.numpy as np
# 
# from operator import itemgetter as itg
# import toolz.curried as z
# from collections import OrderedDict

# ## Load data

# In[ ]:

with open('data/pos.train.txt','r') as f:
    txt = f.read() #


# In[ ]:

sents = filter(None, [zip(*[e.split() for e in sent.splitlines()]) for sent in txt[:].split('\n\n')])
X = map(itg(0), sents)
Y_ = map(itg(1), sents)
tags = sorted({tag for y in Y_ for tag in y if tag.isalpha()})


# In[ ]:

txt[:100]


# In[ ]:

# common bigrams
bigs = defaultdict(lambda: defaultdict(int))

for y in Y_:
    for t1, t2 in zip(y[:-1], y[1:]):
        bigs[t1][t2] += 1
        
bigd = DataFrame(bigs).fillna(0)[tags].ix[tags]
# bigd
# sns.clustermap(bigd, annot=1, figsize=(16, 20), fmt='.0f')


# In[ ]:

wcts_all = defaultdict(Counter)
for xi, yi in zip(X, Y_):
    for xw, yw in zip(xi, yi):
        wcts_all[xw][yw] += 1


# In[ ]:

wcts = z.valfilter(lambda x: sum(x.values()) > 4, wcts_all)


# In[ ]:

' '.join(y)


# In[ ]:

' '.join(tags)


# ## Algo
# 
# $$
# p(\bar y | \bar x;w) =
# \frac {1} {Z(\bar x, w)}
# \exp \sum_j w_j F_j(\bar x, \bar y)
# $$
# 
# $$
# F_j(\bar x, \bar y) = 
# \sum_{i=1}^n f_j(y_{i-1}, y_i, \bar x, i)
# $$

# # Utils Imports

# In[ ]:

import utils; reload(utils); from utils import *
# from utils import sum1, sum2, post_mr, mk_sum, F
fs = AttrDict(fs)
fsums = AttrDict(fsums)


# In[ ]:

def eq(x):
    return lambda y: x == y

def sch(term, x=False):
    "Search in X or Y for term, return matching input and output"
    f = eq(term) if isinstance(term, (str, unicode)) else term
    ss = X if x else Y
    for i, s in enumerate(ss):
        if any(f(t) for t in s):
            yield X[i], Y_[i]


# Yb = map(FeatUtils.mkbookend, Y_)

#     x0 = X[0]
#     y0 = Y_[0]

# ### Argmax
# 
# Get $\text{argmax}_{\bar y} p(\bar y | \bar x;w)$. Since the scoring function only depends on 2 (consecutive in this situation) elements of $\bar y$, argmax can be computed in polynomial time with a table ($\in ℝ^{|Y| \times |y|}$). $U_{ij}$ is the highest score for sequences ending in $y_i$ at position $y_j$.

# $$
# g_i(y_ {i-1}, y_i) = \sum^J_{j=1} w_j f_j (y_ {i-1}, y_i, \bar x, i)
# $$

#     # def gf(ws, yp, y, xbar, i):
#     #     return sum(f(yp, y, xbar, i) * ws[fn] for fn, f in fs.items())
# 
#     x_ = ['Mr.', 'Doo', 'in', 'a', 'circus']
#     y_ = ['NNP', 'NNP', 'IN', 'DT', 'IN']
# 
#     mkwts1 = lambda fs: z.valmap(const(1), fs)
#     ws = mkwts1(fs)
# 
#     gf = mkgf(ws, fs, tags, x_)
#     # gf = mkgf(ws, fs, tags, ['Mr.', 'Happy', 'derp'])

# ### Generate maximum score matrix U

# In[ ]:

def init_u(m0):
    mu = m0.mean()
    ymax = mu.idxmax()
    return ymax, mu[ymax]


def init_score(tags, tag=START, sort=True):
    "Base case for recurrent score calculation U"
    i = Series(0, index=sorted(tags) if sort else tags)
    i.loc[tag] = 1
    return i


#     f = fs2['eq_wd1']
#     f('START', 'TAG1', xt2, 2)
# 
#     F = Fs2['eq_wd2']
#     F(xt2, yt2)

# In[ ]:

def s2df(xs: List[Series]) -> DataFrame:
    return DataFrame({i: s for i, s in enumerate(xs)})

def debugu(ufunc, gmat, uadd, gf, pt, k):
    ufunc.gmat = gmat
    ufunc.uadd = uadd
    pt('\n', k)
    pt(gf.xbar[k], )
    pt(gmat)
    pt('\nuadd')
    pt(uadd)
    
def get_u(k: int=None, gf: "int -> (Y, Y') -> float"=None, collect=True, verbose=False) -> '([max score], [max ix])':
    """Recursively build up g_i matrices bottom up, adding y-1 score
    to get max y score. Returns score.
    - k is in terms of y vector, which is augmented with beginning and end tags
    - also returns indices yprev that maximize y at each level to help reconstruct
        most likely sequence
    """
    pt = testprint(verbose)
    imx = len(gf.xbar) + 1
    if k is None:
        pt(gf.xbar)
        return get_u(imx, gf=gf, collect=1, verbose=verbose)
    if k == 0:
        return [init_score(gf.tags, START)], []

    uprevs, ixprevs = get_u(k - 1, gf=gf, collect=False, verbose=verbose)
    gmat = getmat(gf(k))
    uadd = gmat.add(uprevs[-1], axis='index')
    
    if k > 0:
        # START tag only possible at beginning.
        # There should be a better way of imposing these constraints
        uadd[START] = -1
    if k < imx:
        uadd[END] = -1  # END only possible at the...end
    
    if k == 1:
        idxmax = Series(START, index=gf.tags)  # uadd.ix[START].idxmax()
    else:
        idxmax = uadd.idxmax()
    pt('idxmax:', idxmax, sep='\n')
    retu, reti = uprevs + [uadd.max()], ixprevs + [idxmax]
    if not collect:
        return retu, reti
    return s2df(retu), s2df(reti)


def mlp(idxs, i: int=None, tagsrev: List[Y]=[END]) -> List[Y]:
    "Most likely sequence"
    if i is None:
        return mlp(idxs, i=int(idxs.columns[-1]), tagsrev=tagsrev)
    elif i < 0:
        return tagsrev[::-1]
    tag = tagsrev[-1]
    yprev = idxs.loc[tag, i]
    return mlp(idxs, i=i - 1, tagsrev=tagsrev + [yprev])

# u2, i2 = get_u(gf=test_getu2.gf2, collect=True, verbose=1)
# i2
# u, k = get_u(4, collect=1)


# In[ ]:

def test_getu1(get_u):
    tgs = [START, 'TAG1', END]
    fs = {'eq_wd1': mk_word_tag('wd1', 'TAG1')}
    ytpred = [START, 'TAG1', END]
    x = EasyList(['wd1'])
    
    gf = mkgf(mkwts1(fs), fs, tgs, x)
    u, i = get_u(gf=gf, collect=True)
    assert (u.idxmax() == ytpred).all()
    assert u.iloc[:, -1].max() == 2
    
def test_getu2(get_u):
    tgs = [START, 'TAG1', END]
    x2 = EasyList(['wd1', 'pre-end'])
    fs = {'eq_wd1': mk_word_tag('wd1', 'TAG1'),
          'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END)}
    ws = z.merge(mkwts1(fs), {'pre_endx': 3})
    gf2 = mkgf(ws, fs, tgs, x2)
    assert all(getmat(gf2(3))[END] == 3)
    test_getu2.gf2 = gf2
    test_getu2.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)
    # print(u2)
    assert (u2.idxmax() == [START, 'TAG1', 'TAG1', END]).all()
    assert u2.iloc[:, -1].max() == 5
    assert mlp(i2) == ['START', 'TAG1', 'TAG1', 'END']
    return u2, i2
    
def test_getu3(get_u):
    tgs = [START, 'TAG1', 'PENULTAG', END]
    fs = {'eq_wd1': mk_word_tag('wd1', 'TAG1'),
#           'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END),
          'pre_endy': lambda yp, y, x, i: (yp == 'PENULTAG') and (y == END),
          'start_nonzero': lambda yp, y, x, i: (y == START) and (i != 0),
          'start_zero': lambda yp, y, x, i: (y == START) and (i == 0),
          'end_nonend': lambda yp, y, x, i: (y == END) and (i != (len(x) + 1)),
          'end_end': lambda yp, y, x, i: (y == END) and (i == (len(x) + 1)),
         }
    ws = z.merge(mkwts1(fs), {'pre_endy': 3, 'start_nonzero': -1, 'end_nonend': -1})
    x2 = EasyList(['wd1', 'pre-end', 'whatevs'])
    gf2 = mkgf(ws, fs, tgs, x2)
    test_getu3.gf2 = gf2
    test_getu3.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)
    assert mlp(i2) == ['START', 'TAG1', 'PENULTAG', 'PENULTAG', 'END']
    return u2, i2

test_getu1(get_u)
test_getu2(get_u)
test_getu3(get_u)
None
# u, i = test_getu2()


# In[ ]:

def side_by_side(da, db):
    d = da.copy()
    d2 = DataFrame(db.copy())
    d.columns = pd.MultiIndex.from_product([['A'], list(d)])
    d2.columns = pd.MultiIndex.from_product([['B'], list(d2)])
    d[d2.columns] = d2
    return d

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


# In[ ]:

def predict(xbar=None, fs=None, tags=None, ws=None):
    "Return argmax_y with corresponding score"
    ws = ws or mkwts1(fs)
    gf = mkgf(ws, fs, tags, xbar)
    u, i = get_u(gf=gf, collect=True, verbose=0)
    path = mlp(i)
    return path, u.ix[END].iloc[-1]
    
path2, score2 = predict(xbar=EasyList(['wd1', 'pre-end', 'whatevs']),
                        fs=test_getu3.fs,
                        tags=[START, 'TAG1', 'PENULTAG', END])


# ##Gradient
# $$\frac{\partial}{\partial w_j} \log p(y | x;w) = F_j (x, y) - \frac1 {Z(x, w)} \sum_{y'} F_j (x, y') [\exp \sum_{j'} w_{j'} F_{j'} (x, y')]$$
# $$= F_j (x, y) - E_{y' \sim  p(y | x;w) } [F_j(x,y')]$$
# 

# ## Forward-backward algorithm
# - Partition function $Z(\bar x, w) = \sum_{\bar y} \exp \sum _{j=1} ^ J w_j F_j (\bar x, \bar y) $ can be intractible; forward-backward vectors can make it easier to compute
#    
# $$\alpha (k + 1,v) = \sum_u \alpha (k,u)[\exp g_{k+1}(u,v)] \in ℝ^m$$
# $$\alpha (0,y) = I(y=START)$$
# 
# $$\beta (u, k) = \sum_v [\exp g_{k+1} (u, v)] \beta(v, k+1) $$
# $$\beta (u, n+1) = I(u= END) $$
# 
# Compute partition function $Z$ from either forward or backward vectors
# 
# $$ Z(\bar x, w) = \beta(START, 0) $$
# $$ Z(\bar x, w) = \alpha(n+1, END) $$
# 
# [There seems to be an error in the notes, which state that $Z(\bar x, w) = \sum_v \alpha(n, v) $. If this is the case, $Z$ calculated with $\alpha$ will never get a contribution from $g_{n+1}$, while $Z$ calculated with $\beta$ will in the $\beta(u, n)$ step.]

# In[ ]:

def get_asum(gf, knext=None, vb=False):
    n = len(gf.xbar)
    tags = gf.tags
    p = testprint(vb)
    if knext is None:
        # The first use of the forward vectors is to write
        return get_asum(gf, knext=n+1, vb=vb)
    if knext < 0:
        raise ValueError('k ({}) cannot be negative'.format(k))
    if knext == 0:
        return init_score(tags, tag=START)
    k = knext - 1
    gnext = getmat(gf(knext))
    ak = get_asum(gf, k, vb=vb)
    
    names = 'exp[g{k1}] g{k1} a_{k}'.format(k1=knext, k=k).split()
    p(side_by_side(np.exp(gnext), gnext, ak, names=names))
    return Series([sum([ak[u] * np.exp(gnext.loc[u, v]) for u in tags]) for v in tags], index=tags)


def get_bsum(gf, k=None, vb=False):
    p = testprint(vb)
    n = len(gf.xbar)
    tags = gf.tags
    if k is None:
        return get_bsum(gf, k=0, vb=vb)
    if k > n + 1:
        raise ValueError('{} > length of x {} + 1'.format(k, n))
    if k == n + 1:
        return init_score(gf.tags, tag=END)
    gnext = getmat(gf(k + 1))
    bnext = get_bsum(gf, k + 1, vb=vb)
    p(side_by_side(np.exp(gnext), gnext, bnext, names=['exp[g{}]'.format(k+1), 'g{}'.format(k+1), 'b_{}'.format(k+1)]))
    return Series([sum([np.exp(gnext.loc[u, v]) * bnext[v] for v in tags]) for u in tags], index=tags)

# # get_asum(gf, 3, 1)
# get_asum(gf, 2)
# # get_asum(gf, 1)
# # get_asum(gf, 0, 1)
# get_asum(gf, vb=1)


# In[ ]:

k = 0
tgs = [START, 'TAG1', END]
x = EasyList(['wd1', 'pre-end'])
fs = {
#     'eq_wd1': mk_word_tag('wd1', 'TAG1'),
    'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END)
     }
ws = z.merge(mkwts1(fs), {'pre_endx': 1})
# f = fs['eq_wd1']
# gf = mkgf(ws, fs, tgs, x)
gf = G(fs=fs, tags=tgs, xbar=x, ws=ws)


# In[ ]:


    
get_bsum(gf, 3)
get_bsum(gf, 2)
# get_bsum(gf, 1)
# za = 


# Check correctness of forward and backward vectors.
# - $ Z(\bar x, w) = \beta(START, 0) = \alpha(n+1, END) $
# - For all positions $k=0...n+1$, $\sum_u \alpha(k, u) \beta(u, k) = Z(\bar x, w)$

# In[ ]:

def test_fwd_bkwd():
    tgs = [START, 'TAG1', END]
    x = EasyList(['wd1', 'pre-end'])
    fs = {
#         'eq_wd1': mk_word_tag('wd1', 'TAG1'),
        'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END)
         }
    ws = z.merge(mkwts1(fs), {'pre_endx': 1})
    gf = G(fs=fs, tags=tgs, xbar=x, ws=ws)

    za = get_asum(gf).END
    zb = get_bsum(gf).START
    assert za == zb
    
    for k in range(len(x) + 2):
        assert get_asum(gf, k) @ get_bsum(gf, k) == za
    return za
    
test_fwd_bkwd()


# In[ ]:

gf(3).mat


# In[ ]:

get_asum(gf, 3)


# In[ ]:

get_bsum(gf, 3)


# In[ ]:

get_asum(gf).sum()


# In[ ]:

Series([1, 2, 3]) @ Series([1, 2, 3])


# In[ ]:

get_a(gf)


# In[ ]:

get_asum(gf, k).dot(get_bsum(gf, k))


# In[ ]:

for k in range(4):
    # k = 3
    print(get_asum(gf, k) @ get_bsum(gf, k))
#     print(get_a.a(k) @ get_b.b(k))


# In[ ]:

get_a(gf, 2)


# In[ ]:

get_a(gf), get_b(gf)


# In[ ]:

get_a(gf).sum()


# In[ ]:





# In[ ]:

get_b.b(k)


# In[ ]:

get_a.a(k)


# In[ ]:

get_b(gf)


# In[ ]:

a1 = 
gf


# In[ ]:

gfa = mkgf(ws, fs, tags, xx)
g1 = gfa(1)


# In[ ]:

del forwarder


# In[ ]:

yt = 


# In[ ]:

fs


# In[ ]:

xt = EasyList(['wd1', 'wd2'])
xt


# In[ ]:

def mkforward(i=0, aprevs=None):
    if not i:
        return mkforward(i=i + 1, aprevs=[mka(y[i] == V)])
    if i >= len(y):
        return aprevs
    aprev = aprevs[-1]
    gk = mkg(i)
    ai = mka([sum(aprev[u] * np.e ** gk(u, v) for u in V) for v in V])
    return mkforward(i=i + 1, aprevs=aprevs + [ai])


# In[ ]:

def alpha(x, y, V, ws, fs) -> List[Series]:
    """Unnormalized probability of set of possible sequences that end at position
    `col` with tag `row`
    """
    mka = lambda x: Series(list(x), index=V)
    mkg = mkgf(ws, fs, V, x)
    def mkforward(i=0, aprevs=None):
        if not i:
            return mkforward(i=i + 1, aprevs=[mka(y[i] == V)])
        if i >= len(y):
            return aprevs
        aprev = aprevs[-1]
        gk = mkg(i)
        ai = mka([sum(aprev[u] * np.e ** gk(u, v) for u in V) for v in V])
        return mkforward(i=i + 1, aprevs=aprevs + [ai])
    return DataFrame(mkforward()).T


# In[ ]:

def forward(x, y, V, ws, fs) -> List[Series]:
    """Unnormalized probability of set of possible sequences that end at position
    `col` with tag `row`
    """
    mka = lambda x: Series(list(x), index=V)
    mkg = mkgf(ws, fs, V, x)
    def mkforward(i=0, aprevs=None):
        if not i:
            return mkforward(i=i + 1, aprevs=[mka(y[i] == V)])
        if i >= len(y):
            return aprevs
        aprev = aprevs[-1]
        gk = mkg(i)
        ai = mka([sum(aprev[u] * np.e ** gk(u, v) for u in V) for v in V])
        return mkforward(i=i + 1, aprevs=aprevs + [ai])
    return DataFrame(mkforward()).T


# In[ ]:

z.operator.sub(1)(9)


# In[ ]:

def backward(x, y, V, ws, fs) -> List[Series]:
    mksrs = lambda x: Series(list(x), index=V)
    mkg = mkgf(ws, fs, V, x)
    i_init = len(y)-1
    i_fin = -1
    nxt = lambda x: x - 1
    
    def mkprobvec(i=i_init, pprevs=None):
        if i == i_init:
            return mkprobvec(i=nxt(i), pprevs=[mksrs(y[i] == V)])
        if i == i_fin:
            return pprevs
        pprev = pprevs[-1]
        gk = mkg(i)
        ai = mksrs([sum(pprev[inv] * np.e ** gk(outv, inv) for inv in V) for outv in V])
        return mkprobvec(i=nxt(i), pprevs=pprevs + [ai])
    return DataFrame(mkprobvec()[::-1]).T


# In[ ]:

DataFrame([xx, yy])


# In[ ]:

aa


# In[ ]:

bb = backward(xx, yy, ts, ws, fs)
bb


# In[ ]:

yy


# In[ ]:

aa = forward(xx, yy, ts, ws, fs)
aa.iloc[:, -1].sum()


# In[ ]:

bb.sum().sum()


# In[ ]:

[sum(a0[u] * np.e ** g1(u, v) for u in ts) for v in ts]


# In[ ]:

[np.e ** sum(g1(u, v) for u in ts) for v in ts]


# In[ ]:




# In[ ]:

fsums.wd_a(xx, yy)


# $$U(k, v) = \max_u [U(k-1, u) + g_k(u,v)]$$
# $$U(1, vec) = \max_{y_0} [U(0, y_0) + g_k(y_0,vec)]$$

# ## Extra

# In[ ]:

t = '''$3,275 Individual $6,550 Family
$2,000 Individual $4,000 Family
$2,000 Individual $4,000 Family
'''.splitlines()
x = map(str.split, t)
i = z.pipe(x, z.map(itg(0)), '\t'.join)
# i = map(itg(0), x)
# f = map(itg(2), x)
f
i


# In[ ]:



