
# coding: utf-8

# wo crf

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
from typing import List, Dict, Tuple
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

import utils; reload(utils); from utils import *
# from utils import sum1, sum2, post_mr, mk_sum, F
fs = AttrDict(fs)
fsums = AttrDict(fsums)


# In[ ]:

sents = filter(None, [zip(*[e.split() for e in sent.splitlines()]) for sent in txt[:].split('\n\n')])
X = map(itg(0), sents)
Y_ = map(itg(1), sents)
Xa = map(EasyList, X)
Ya = map(AugmentY, Y_)
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
# $$U(k, v) = \max_u [U(k-1, u) + g_k(u,v)]$$
# $$U(1, vec) = \max_{y_0} [U(0, y_0) + g_k(y_0,vec)]$$

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


# In[ ]:

from test import no_test_getu1, no_test_getu2, no_test_getu3

no_test_getu1(get_u, mlp)
no_test_getu2(get_u, mlp)
no_test_getu3(get_u, mlp)
None


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
                        fs=no_test_getu3.fs,
                        tags=[START, 'TAG1', 'PENULTAG', END])


# ##Gradient
# $$\frac{\partial}{\partial w_j} \log p(y | x;w) = F_j (x, y) - \frac1 {Z(x, w)} \sum_{y'} F_j (x, y') [\exp \sum_{j'} w_{j'} F_{j'} (x, y')]$$
# $$= F_j (x, y) - E_{y' \sim  p(y | x;w) } [F_j(x,y')]$$
# 
# 
# ### Forward-backward algorithm
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
def get_bsum(gf, k=None, vb=False):


# Check correctness of forward and backward vectors.
# - $ Z(\bar x, w) = \beta(START, 0) = \alpha(n+1, END) $
# - For all positions $k=0...n+1$, $\sum_u \alpha(k, u) \beta(u, k) = Z(\bar x, w)$

# In[ ]:

def test_fwd_bkwd():
    tgs = [START, 'TAG1', END]
    x = EasyList(['wd1', 'pre-end'])
    fs = {
        # 'eq_wd1': mk_word_tag('wd1', 'TAG1'),
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


# ### Calculate expected value of feature function
# Weighted by conditional probability of $y'$ given $x$
# $$E_{y' \sim  p(y | x;w) } [F_j(x,y')]$$

# # Utils Imports

# In[ ]:

import utils; reload(utils); from utils import *
# from utils import sum1, sum2, post_mr, mk_sum, F
fs = AttrDict(fs)
fsums = AttrDict(fsums)


# In[ ]:

k = 0
tgs = [START, 'TAG1', END]
x = EasyList(['wd1', 'pre-end'])
fs = {
#     'eq_wd1': mk_word_tag('wd1', 'TAG1'),
    'pre_endx': lambda yp, y, x, i: (x[i - 1] == 'pre-end') and (y == END)
     }
yb = ['TAG1', 'TAG1']
ybar = AugmentY(yb)
ws = z.merge(mkwts1(fs), {'pre_endx': 1})
# f = fs['eq_wd1']
# gf = mkgf(ws, fs, tgs, x)
gf = G(fs=fs, tags=tgs, xbar=x, ws=ws)


# In[ ]:

fj = fs['pre_endx']


# In[ ]:

def expectation(gf, fj):
    n = len(gf.xbar)
    ss = 0
    za = get_asum(gf).END
    
    for i in range(1, n + 2):
        gfix = np.exp(gf(i).mat)
        alpha_vec = get_asum(gf, i - 1)
        beta_vec = get_bsum(gf, i)
        ss += sum(
                [fj(yprev, y, gf.xbar, i) * alpha_vec[yprev] * gfix.loc[yprev, y] * beta_vec[y]
                for yprev in tgs
            for y in tgs])
    return ss / za

def expectation2(gf, fj):
    n = len(gf.xbar)
    ss = 0
    za = get_asum(gf).END
    
    for i in range(1, n + 2):
        gfix = np.exp(gf(i).mat)
        alpha_vec = get_asum(gf, i - 1)
        beta_vec = get_bsum(gf, i)
        for yprev in tgs:
            for y in tgs:
                ff = fj(yprev, y, gf.xbar, i)
                α = alpha_vec[yprev]
                β = beta_vec[y]
                gfx = gfix.loc[yprev, y]
                ss += ff * α * β * gfx
    return ss / za

ee = 1

# %time expectation(gf, fs['pre_endx'])
# e1, e2


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
    if vb:
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
    if vb:
        p(side_by_side(np.exp(gnext), gnext, bnext, names=['exp[g{}]'.format(k+1), 'g{}'.format(k+1), 'b_{}'.format(k+1)]))
    return Series([sum([np.exp(gnext.loc[u, v]) * bnext[v] for v in tags]) for u in tags], index=tags)


# In[ ]:

get_ipython().magic('prun -qD profex.prof expectation2(gf, fj)')


# In[ ]:

expectation(gf, fj)


# In[ ]:

expectation2(gf, fj)


# In[ ]:

fj


# In[ ]:

ag = ybar.aug
yn = len(ag)
nmx = yn - 1


# In[ ]:

fj(ag[nmx - 1], ag[nmx], x, nmx)


# In[ ]:

ag


# In[ ]:

Fj = FeatUtils.mk_sum(fj)
Fj(x, yb)


# In[ ]:

def partial_d(gf, fj, x, y, Fj=None) -> float:
    if Fj is None:
        Fj = FeatUtils.mk_sum(fj)
    return Fj(x, y) - expectation(gf, fj)


# ## Test Partial

# In[ ]:

def process_corpus(corpus, sep='//'):
    psplit = lambda f: (lambda xs: map(z.comp(f, str.split), xs))
    linepairs = z.comp(z.map(mc('split', sep)), str.splitlines)(corpus)
    xs_, ys_ = zip(*linepairs)
    xs, ys = psplit(EasyList)(xs_), psplit(AugmentY)(ys_)
    return xs, ys 


# In[ ]:




# In[ ]:

Xa[:2]
Ya[:2]


# In[ ]:

corpus = '''Nothing seems hard here .//NN VBZ JJ RB .
The reason is cost .//DT NN VBZ NN .
Terms were n't disclosed .//NNS VBD RB VBN .
Mr. Juliano really really thinks so .//NNP NNP RB RB VBZ RB .
Mr. Bill seems dead .//NNP NNP VBZ JJ .
Young & Rubicam 's Pact//NNP CC NNP POS NNP
Albany escaped embarrassingly unscathed .//NNP VBD RB JJ .'''


# In[ ]:

xs, ys = process_corpus(corpus)
zs = zip(xs, ys)
tgs = sorted({y for ybar in ys for y in ybar.aug})


iscapped = lambda x: x and x[0].isupper()


# In[ ]:

def mk_fx_tag(fx, tag):
    def f(yp_, y, x, i):
        return x[i] and fx(x[i]) and (y == tag)
    f.__name__ = '{}(x)_{}'.format(fx, tag)
    f.__doc__ = '{}(x[i]) and (y == {})'.format(fx, tag)
    return f

def runFs(Fj, zs=zs):
    return [Fj(x, y) for x, y in zs]


# In[ ]:

ys


# In[ ]:

tgs


# In[ ]:

fs = dict(
    seems_VBZ=mk_word_tag('seems', 'VBZ'),
    ly_VBZ=lambda yp, y, x, i: x[i] and x[i].endswith('ly') and (y == 'RB'),
    cap_NN=mk_fx_tag(iscapped, 'NN'),
    cap_NNP=mk_fx_tag(iscapped, 'NNP'),
    nocap_START=lambda yp, y, x, i: x[i] and not iscapped(x[i]) and (yp == START),
#     cap_NN=lambda yp, y, x, i: iscapped(x[i]) and (y == 'NN'),
)
Fs = z.valmap(FeatUtils.mk_sum, fs)

assert sum(runFs(Fs['ly_VBZ'])) == 3
assert sum(runFs(Fs['cap_NNP'])) == 8
assert sum(runFs(Fs['cap_NN'])) == 1
assert not sum(runFs(Fs['nocap_START']))


# In[ ]:

d = Derp()
d + 1
d * 3
1 + d


# In[ ]:

FeatUtils.bookend = False


# In[ ]:


# Fj = FeatUtils.mk_sum(ly)


# In[ ]:

get_ipython().magic('prun -qD prof.prof partial_d(gf, fj, x, y, Fj=None)')


# In[ ]:

λ = 1
α


# In[ ]:




# In[ ]:

fj = fs['ly_VBZ']
Fj = Fs['ly_VBZ']
ws = mkwts1(fs, const=1)

for x, y in zs:
    gf = G(fs=fs, tags=tgs, xbar=x, ws=ws)
    if not Fj(x, y):  # TODO: is this always right?
        continue
    pder = partial_d(gf, fj, x, y, Fj=Fj)
    wj0 = ws['ly_VBZ']
    ws['ly_VBZ'] += λ * pder
    print('wj: {} -> {}'.format(wj0, ws['ly_VBZ']))
    print('pder: {:.2f}'.format(pder), Fj(x, y))
    


# In[ ]:

zs


# In[ ]:




# In[ ]:

del Fj


# In[ ]:

fj = fs['ly_VBZ']
# Fj = Fs['ly_VBZ']

def train_(zs: List[Tuple[EasyList, AugmentY]],
          fjid='ly_VBZ', ws=ws, vb=True, tgs=tgs):
    fj = fs[fjid]
    Fj = FeatUtils.mk_sum(fj)
    pt = testprint(vb)
    for x, y in zs:
        gf = G(fs=fs, tags=tgs, xbar=x, ws=ws)
        if not Fj(x, y):  # TODO: is this always right?
            continue
        pder = partial_d(gf, fj, x, y, Fj=Fj)
        wj0 = ws[fjid]
        ws[fjid] += λ * pder
        pt('wj: {} -> {}'.format(wj0, ws[fjid]))
        pt('pder: {:.2f}'.format(pder), Fj(x, y))
    return ws

def train_j(zs: List[Tuple[EasyList, AugmentY]],
          fjid='ly_VBZ', ws=ws, tol=.001, maxiter=10, vb=True, tgs=tgs):
    ws1 = ws
    pt = testprint(vb)
    
    for i in count(1):
        pt('Iter', i)
        wj1 = ws1[fjid]
        ws2 = train_(zs, fjid=fjid, ws=ws1, vb=vb, tgs=tgs)
        wj2 = ws2[fjid]
        if abs((wj2 - wj1) / wj1) < tol or (i >= maxiter):
            return ws, i
        ws1 = ws2
        
def train(zs, fs, ws, tol=.001, maxiter=10, vb=False, tgs=tgs):
    wst = ws.copy()
    for fname, f in fs.items():
        wst, i = train_j(zs, fjid=fname, ws=wst, tol=tol, maxiter=maxiter, vb=vb, tgs=tgs)
        print(fname, 'trained in', i, 'iters: {:.2f}'.format(wst[fname]))
        sys.stdout.flush()
    return wst


# In[ ]:

train_j(zs, fjid='ly_VBZ', ws=ws)


# In[ ]:

get_ipython().magic('time ws1 = train(zs, fs, mkwts1(fs, 1), maxiter=1)')


# In[ ]:

# ws3 = train(zs, fs, mkwts1(fs, 1))
ws3 = train(zs, fs, ws3, vb=False, maxiter=20, tol=.01)


# In[ ]:

ws3


# In[ ]:

ws2


# In[ ]:




# In[ ]:

ws2 = train(zs, fjid='ly_VBZ', ws=ws)


# In[ ]:

ws2


# In[ ]:

ws3 = train(zs, fjid='ly_VBZ', ws=ws2)


# In[ ]:

ws3


# In[ ]:

ws2 = ws.copy


# In[ ]:

gf.ws['ly_VBZ']


# In[ ]:

gf.ws['ly_VBZ'] -= 1


# In[ ]:

gf._replace(ws=z.valmap(lambda x: x + 1, ws))


# In[ ]:

gf.ws = ws


# In[ ]:

for Fj 


# In[ ]:

fs


# In[ ]:




# In[ ]:

xs, ys


# In[ ]:

fives = [(x, y) for x, y in zip(X, Y_) if len(x) == 5]
fives = DataFrame(fives).applymap(' '.join)
fives


# In[ ]:

Series(map(len, X)).value_counts(normalize=0)


# In[ ]:

x


# In[ ]:

ybar


# In[ ]:

Fj(x, ybar)


# In[ ]:

partial_d(gf, fj, x, ybar)


# In[ ]:




# In[ ]:

AugmentY(AugmentY(yb))


# In[ ]:

yb


# In[ ]:

ss


# ## Extra
