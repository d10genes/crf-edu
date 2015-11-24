
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
from typing import List
Y = str

if sys.version_info.major > 2:
    unicode = str


# from py3k_imports import *
# from project_imports3 import *
# import pandas as pd
# import seaborn as sns
# import autograd.numpy as np
# 
# from operator import itemgetter as itg
# import toolz.curried as z
# from collections import OrderedDict

# # Viterbi
# From [here](http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf)
# 
# s = StringIO()
# states2.to_csv(s)
# s.seek(0)
# sc = s.read()

#     from io import StringIO
#     csvstr = ''',H,L
#     A,0.2,0.3
#     C,0.3,0.2
#     G,0.3,0.2
#     T,0.2,0.3'''
#     states = pd.read_csv(StringIO(csvstr), index_col=0).T
# 
#     sts = list(states.index)
#     start = [.5, .5]
#     transition = DataFrame([[.5, .5], [.4, .6]], columns=Series(sts, name='To'), index=Series(sts, name='From'))
# 

# In[ ]:

start = [.5, .5]
state_names = ['H', 'L']
obs_names = list('ACGT')
transition = DataFrame([[.5, .5], [.4, .6]],
                       columns=Series(state_names, name='To'),
                       index=Series(state_names, name='From'))
states = DataFrame(
    [[0.2, 0.3, 0.3, 0.2],
     [0.3, 0.2, 0.2, 0.3]], index=state_names, columns=obs_names)

log = np.log2
startL, transitionL, statesL = map(log, [start, transition, states])


# In[ ]:

def build_table(s, startL, statesL, transitionL, convert_log=False):
    """
    s: observed sequence
    startL: initial state probabilities
    statesL: p(obs_j | hidden state_i)
    transitionL: p(transition to state_j|state_i)
    
    """
    if convert_log:
        startL, statesL, transitionL = map(np.log, [startL, statesL, transitionL])
    prev_path = DataFrame()
    probs = DataFrame({0: startL + statesL[s[0]]})

    for i, l in enumerate(s[1:], 1):
        tocur = transitionL.add(probs[i-1], axis='index')  # p(current_state | each possible prev. state)
        prev_path[i-1] = tocur.idxmax()
        probs[i] = tocur.max() + statesL[l]  # p(current_state | most likely prev. state) * p(z|current_state)
    return probs, prev_path

def most_likely_path(probs, prev_path):
    final_likely_state = probs.iloc[:, -1].idxmax()
    backwards_path = [final_likely_state]

    for c in reversed(list(prev_path)):
        backwards_path.append(prev_path[c].ix[backwards_path[-1]])
    mlp = backwards_path[::-1]
    return mlp

probs, prev_path = build_table('GGCACTGAA', startL, statesL, transitionL)
mlp = most_likely_path(probs, prev_path)
mlp


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


#     Series(Counter(y[0] for y in Y)).order(ascending=0)[:5]
#     Series(Counter(y[:2][-1] for y in Y)).order(ascending=0)[:4]
# 
#     Series(Counter(y[-2:][0] for y in Y)).order(ascending=0)[:4]

# In[ ]:

wcts_all = defaultdict(Counter)
for xi, yi in zip(X, Y_):
    for xw, yw in zip(xi, yi):
        wcts_all[xw][yw] += 1


# In[ ]:

wcts = z.valfilter(lambda x: sum(x.values()) > 4, wcts_all)


#     stops = 'the of to a and in for that'.split()
#     stops = Series(stops)
#     z.reduceby()
#     def get_max(d):
#         k, v = max(d.items(), key=snd)
#         return k, v / sum(d.values())
# 
#     get_max({'IN': 17, 'DT': 9202})
#     s = stops.map(z.comp(get_max, wcts.get)).reset_index(drop=0).set_index(stops)[0].order(ascending=0).drop('that', axis=0)
#     s
#     s.index

#     cts = Series(z.valmap(lambda x: sum(x.values()), wcts))
#     cts.value_counts(normalize=1)
#     cts.order(ascending=0)

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

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# # Imports

# In[ ]:

import utils; reload(utils); from utils import *
# from utils import sum1, sum2, post_mr, mk_sum, F
fs = AttrDict(fs)
fsums = AttrDict(fsums)


# In[ ]:

def eq(x):
    return lambda y: x == y

def sch(term, x=False):
    f = eq(term) if isinstance(term, (str, unicode)) else term
    ss = X if x else Y
    for i, s in enumerate(ss):
        if any(f(t) for t in s):
            yield X[i], Y_[i]


# Yb = map(FeatUtils.mkbookend, Y_)

# In[ ]:

x0 = X[0]
y0 = Y_[0]


# ### Argmax
# $\newcommand{\argmin}{\operatornamewithlimits{argmin}}$
# Get $\text{argmax}_{\bar y} p(\bar y | \bar x;w) $
# 
# 

# In[ ]:

# ws = np.ones_like(fs.values())
ws = z.valmap(const(1), fs)


# $$
# g_i(y_ {i-1}, y_i) = \sum^J_{j=1} w_j f_j (y_ {i-1}, y_i, \bar x, i)
# $$

# In[ ]:

Df = Dict


# In[ ]:

# def gf(ws, yp, y, xbar, i):
#     return sum(f(yp, y, xbar, i) * ws[fn] for fn, f in fs.items())

xx = ['Mr.', 'Doo', 'in', 'a', 'circus']
yy = ['NNP', 'NNP', 'IN', 'DT', 'IN']
gf = mkgf(ws, fs, tags, xx)
# gf = mkgf(ws, fs, tags, ['Mr.', 'Happy', 'derp'])


#     test_mats()
#     gft = test_mats_2_args()
#     gf0 = gft(0)
#     gf1 = gft(1)
# 
#     m0 = getmat(gft(0))
#     m1 = getmat(gft(1))
# 
#     gf1 = gf(1)
#     gf0 = gf(0)
#     gf1

# ### Generate maximum score matrix U

# In[ ]:

def init_u(m0):
    mu = m0.mean()
    ymax = mu.idxmax()
    return ymax, mu[ymax]

def init_score(tags, tag=START):
    "Base case for recurrent score calculation U"
    i = Series(0, index=tags)
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
    
def get_u(k: int=None, gf: "int -> (Y, Y') -> float"=gf, collect=True, verbose=False) -> '([max score], [max ix])':
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
    if k == imx and 0:
        gmat = gmat[[END]]
    uadd = gmat.add(uprevs[-1], axis='index')
    if k > 0:
        uadd[START] = -1  # START only possible at beginning
    if k < imx:
        uadd[END] = -1  # START only possible at beginning
    
    debugu(get_u, gmat, uadd, gf, pt, k)
    if k == 1:
        idxmax = Series(START, index=gf.tags)  # uadd.ix[START].idxmax()
    else:
        idxmax = uadd.idxmax()
    pt('idxmax:', idxmax, sep='\n')
    retu, reti = uprevs + [uadd.max()], ixprevs + [idxmax]
    if not collect:
        return retu, reti
    return s2df(retu), s2df(reti)


# INIT = object()
def mlp(idxs, i: int=None, tagsrev: List[Y]=[END]) -> List[Y]:
    if i is None:
        return mlp(idxs, i=int(idxs.columns[-1]), tagsrev=tagsrev)
    elif i < 0:
        return tagsrev[::-1]
    tag = tagsrev[-1]
    yprev = idxs.loc[tag, i]
    # u.iloc[:, -1][tag]
    
    return mlp(idxs, i=i - 1, tagsrev=tagsrev + [yprev])

# u2, i2 = get_u(gf=test_getu2.gf2, collect=True, verbose=1)
# i2
# u, k = get_u(4, collect=1)


# In[ ]:

def test_getu1():
    tgs = [START, 'TAG1', END]
    fs = {'eq_wd1': mk_word_tag('wd1', 'TAG1')}
    ytpred = [START, 'TAG1', END]
    x = EasyList(['wd1'])
    
    gf = mkgf(mkwts1(fs), fs, tgs, x)
    u, i = get_u(gf=gf, collect=True)
    assert (u.idxmax() == ytpred).all()
    assert u.iloc[:, -1].max() == 2
    
def test_getu2():
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
    print(u2)
    assert (u2.idxmax() == [START, 'TAG1', 'TAG1', END]).all()
    assert u2.iloc[:, -1].max() == 5
    assert mlp(i2) == ['START', 'TAG1', 'TAG1', 'END']
    return u2, i2
    
test_getu1()
u, i = test_getu2()


# In[ ]:

def test_getu3():
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
#     print(getmat(gf2(3)))
    # assert all(getmat(gf2(3))[END] == 3)
    test_getu3.gf2 = gf2
    test_getu3.fs = fs
    u2, i2 = get_u(gf=gf2, collect=True, verbose=0)
#     assert (u2.idxmax() == [START, 'TAG1', END, END]).all()
#     # 3rd value for predicted sequence is END, but only because it is first in index order
#     assert u2[2].nunique() == 1, '3rd predicted tag should have same score for all v`s'
#     assert u2.iloc[:, -1].max() == 5
    assert mlp(i2) == ['START', 'TAG1', 'PENULTAG', 'PENULTAG', 'END']
    return u2, i2

u, i = test_getu3()
g = test_getu3.gf2
fs = test_getu3.fs
f = fs['start_nonzero']
# i
x2 = EasyList(['wd1', 'pre-end', 'whatevs'])
u


# In[ ]:

def side_by_side(da, db):
    d = da.copy()
    d2 = DataFrame(db.copy())
    d.columns = pd.MultiIndex.from_product([['A'], list(d)])
    d2.columns = pd.MultiIndex.from_product([['B'], list(d2)])
    d[d2.columns] = d2
    return d

def side_by_side(*ds):
    dmultis = [side_by_side1(d, ctr=i) for i, d in enumerate(ds)]
    return pd.concat(dmultis, axis=1)

def side_by_side1(d, ctr=1):
    d = DataFrame(d.copy())
    d.columns = pd.MultiIndex.from_product([[ctr], list(d)])
    return d
    

def side_by_side_(*objs, **kwds):
    from pandas.core.common import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))
def ff(m):
    return side_by_side(m, m.idxmax(), m.max())


# In[ ]:

# Only need to keep max and idxmax at each level
# y0 at level1 does not affect y2 at level 2 (but y1 will!)


# In[ ]:

u2 = getmat(g(2))
u2 = u2.add(u1.T, axis='index')
u2i, u2m = u2.idxmax(), u2.max()
side_bu_side(u2, u2i, u2m)


# In[ ]:

u3 = getmat(g(3))
ff(u3)


# In[ ]:

u3.add(u2m, axis='index')


# In[ ]:

u2.add(u1.T, axis='index')


# In[ ]:

s2df(i)


# # Import2

# In[ ]:

import utils; reload(utils); from utils import *
# from utils import sum1, sum2, post_mr, mk_sum, F
fs = AttrDict(fs)
fsums = AttrDict(fsums)


#     def post_mr(yp, y, x, i):  # optional keywords to not confuse mypy
#         return (y == yp == 'NNP') & (x[i - 1] == 'Mr.')
# 
#     last_nn = lambda yp_, y, x, i: (i == len(x) - 1) and (y == 'NN')
#     last_nn = lambda yp, y, x, i: (yp == 'NNP') and (y == END)
# 
#     F = mk_sum(post_mr)
#     # F(['wd0', 'Mr.', 'pre-end'], ['TAG3', 'NNP', 'NNP'])
#     F = mk_sum(last_nn)
#     F(['to', '1.23', 'the'], ['TO', 'CD', 'NN'])
#     xt2
#     yt2
# 
#     def test_likely_path():
#         testfs_str = 'wd_to wd_of wd_for wd_in wd_a wd_the wd_and'.split()
#         testfs = z.keyfilter(lambda x: x in testfs_str, fs)
#         wst = mkwts1(testfs)
#         stags = 'Junk1 Junk2 TO IN DT CC Junk3 Junk4'.split()
# 
#         xt = 'of for in the and a to'.split()
#         gft = mkgf(wst, testfs, stags, xt)
#         # gft = mkgf(wst, testfs, stags, xt)
#         u, i = get_u(gf=gft, collect=True)
# 
#         shouldbe_dims = len(stags), len(xt)
#         assert u.shape == shouldbe_dims, ('Shape of score matrix is wrong. '
#                                           'Actual: {}, Expected: {}'.format(u.shape, shouldbe_dims))
#         assert most_likely_path(u, i) == [None, 'IN', 'IN', 'IN', 'DT', 'CC', 'DT', 'TO']
# 
#     test_likely_path()
# 
#     for i in range(1, len(yt2)):
#         print(i, end=' ')
#         print(yt2[i])
# 
#     xt = 'Hi this has Two capped words'.split()
#     resmat = getmat(gft(0))
#     gft = mkgf()
# 
#     def most_likely_path(u: 'DataFrame[float]', i: 'DataFrame[Y]') -> (List[str], List[float]):
#         revpath = []
#         revscore = []
# 
#         for c in reversed(u.columns[:]):
#     #         print(c)
#             ix = u[c].idxmax()
#             revscore.append(u[c][ix])
#             revpath.append(ix)
#             prevmax = i[c][ix]
#     #         print('ix:', ix)
#     #         print('prevmax:', prevmax)
#             if c:
#                 assert u[c-1].max() == u[c-1][prevmax]
#         #     break
#         return revpath[::-1], revscore[::-1]

# In[ ]:

# path, score = most_likely_path(uu, ii)
path, score = most_likely_path(u, i)


# In[ ]:

path


# In[ ]:

def predict(xbar, ws, fs, tags):
    gf = mkgf(ws, fs, tags, xbar)
    u, i = get_u(len(xbar) - 1, gf=gf, collect=True)
#     return u, i
#     print(u)
    path, score = most_likely_path(u, i)
    return path, score
    
path2, score2 = predict(['Mr.', 'Doo', 'is', 'in', 'a', 'circus'], ws, fs, tags)


# In[ ]:

predict(['Mr.', 'Doo', 'in', 'a', 'circus'], ws, fs, tags)


# ##Gradient
# $$\frac{\partial}{\partial w_j} \log p(y | x;w) = F_j (x, y) - E_{y' \sim  p(y | x;w) } [F_j(x,y')]$$
# 
# $$\alpha (0,y) = I(y=start)$$
# $$\alpha (k + 1,v) = \sum_u \alpha (k,u)[\exp g_{k+1}(u,v)] \in ℝ^m$$

# In[ ]:

ts = Series(tags)
y = yy
y


# In[ ]:

V = ts
v0 = v[0]
v0


# In[ ]:

k = 0
# vi = 
y0 = y[0]
a0 = Series(list(y0 == ts), index=ts)
# a0.reset_index(drop=0)[:15].T.ix[[0]]
a0


# In[ ]:

gfa = mkgf(ws, fs, tags, xx)
g1 = gfa(1)


# In[ ]:

del forwarder


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



