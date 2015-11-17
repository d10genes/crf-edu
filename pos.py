
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


# In[ ]:

import utils; reload(utils)
from utils import *
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


# In[ ]:

Yb = map(FeatUtils.mkbookend, Y_)


# In[ ]:

enumx = lambda x: range(1, len(x) + 1)
enumxy = lambda x: enumerate(range(1, len(x) + 1))

def test_enumxy():
    x = [1, 2, 3, 4]
    y = ['START', 1, 2, 3, 4, 'END']
    assert all([(x[i] == y[j]) for i, j in enumxy(x)])
    assert [y[i] for i in enumx(x)] == x

test_enumxy()


# In[ ]:

# g = sch('Mr.', 1)
# x, y = g.__next__()


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

# def gf(ws, yp, y, xbar, i):
#     return sum(f(yp, y, xbar, i) * ws[fn] for fn, f in fs.items())

def mkgf(ws, fs, tags, xbar):
    #@z.curry
    def gf(i):
        def gfi(yp, y):
            return sum(f(yp, y, xbar, i) * ws[fn] for fn, f in fs.items())
        gfi.tags = tags
        return gfi
    gf.xbar = xbar
    return gf

def getmat(gf):
    df = DataFrame({ytag: {ytag_prev: gf(ytag_prev, ytag) for ytag_prev in gf.tags}
                    for ytag in gf.tags})
    df.columns.name, df.index.name = 'Y', 'Yprev'
    return df

xx = ['Mr.', 'Doo', 'in', 'a', 'circus']
yy = ['NNP', 'NNP', 'IN', 'DT', 'IN']
gf = mkgf(ws, fs, tags, xx)
# gf = mkgf(ws, fs, tags, ['Mr.', 'Happy', 'derp'])


# In[ ]:

def test_mats():
    xt = 'Hi this has Two capped words'.split()
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    testfs = dict(cap_nnp=fs['cap_nnp'])
    gft = mkgf(wst, testfs, stags, xt)
    resmat = getmat(gft(0))

    assert all(resmat.NNP == 1)
    assert (resmat.drop('NNP', axis=1) == 0).all().all()
    return 0

def test_mats_2_args():
    xt = 'Mr. Derp has Three capped words'.split()
    testfs = z.keyfilter(lambda x: x in ('cap_nnp', 'post_mr'), fs)
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

    gft = mkgf(wst, testfs, stags, xt)
    m0 = getmat(gft(0))
    m1 = getmat(gft(1))

    # First position should be the same
    assert all(m0.NNP == 1)
    assert (m0.drop('NNP', axis=1) == 0).all().all()
    
    # Second should get additional point from Mr. feature in position
    # y-1 == NNP, y == NNP
    assert m1.NNP.NNP == 2
    # Subtracting that should give same matrix as original
    m1c = m1.copy()
    m1c.loc['NNP', 'NNP'] = m1.NNP.NNP - 1
    assert_frame_equal(m1c, m0)
    return gft
    return 0

test_mats()
gft = test_mats_2_args()
gf0 = gft(0)
gf1 = gft(1)

m0 = getmat(gft(0))
m1 = getmat(gft(1))


# In[ ]:

gf1 = gf(1)
gf0 = gf(0)
gf1


# ### Generate maximum score matrix U

# In[ ]:

def init_u(m0):
    mu = m0.mean()
    ymax = mu.idxmax()
    return ymax, mu[ymax]


# In[ ]:

mu = m0.mean()
ymax = mu.idxmax()
mu[ymax]


# In[ ]:

init_u(m0)


# In[ ]:

u0 = getmat(gf(0)).mean()
u0.iloc[:7]


# In[ ]:

def s2df(xs: List[Series]) -> DataFrame:
    return DataFrame({i: s for i, s in enumerate(xs)})

def get_u(i: int=None, gf: "int -> (Y, Y') -> float"=gf, collect=True) -> '([max score], [max ix])':
    """Recursively build up g_i matrices bottom up, adding y-1 score
    to get max y score. Returns score
    """
    if i is None:
        return get_u(i=len(gf.xbar) - 1, gf=gf, collect=collect)
    gmat = getmat(gf(i))
    if not i:
        return [gmat.mean()], [None]
    uprevs, ixprevs = get_u(i - 1, gf=gf, collect=False)
    uadd = gmat.add(uprevs[-1], axis='index')
    retu, reti = uprevs + [uadd.max()], ixprevs + [uadd.idxmax()]
    if not collect:
        return retu, reti
    return s2df(retu), s2df(reti)
    
u, i = get_u(4, collect=1)


# In[ ]:

ut, it = get_u(gf=gft, collect=True)
ut


# In[ ]:

mk_word_tag


# In[ ]:

fs2 = [('wd1', 'TAG1'), ('wd', 'TAG'), ('wd3', 'TAG3'), ('wd4', 'TAG4'),
      (lambda yp_, y, x, i: ()]
fs2


# In[ ]:

mk_word_tag('wd1', 'TAG1')


# In[ ]:

def test_likely_path():
    testfs_str = 'wd_to wd_of wd_for wd_in wd_a wd_the wd_and'.split()
    testfs = z.keyfilter(lambda x: x in testfs_str, fs)
    wst = z.valmap(const(1), testfs)
    stags = 'Junk1 Junk2 TO IN DT CC Junk3 Junk4'.split()
    
    xt = 'of for in the and a to'.split()
    gft = mkgf(wst, testfs, stags, xt)
    # gft = mkgf(wst, testfs, stags, xt)
    u, i = get_u(gf=gft, collect=True)
    
    shouldbe_dims = len(stags), len(xt)
    assert u.shape == shouldbe_dims, ('Shape of score matrix is wrong. '
                                      'Actual: {}, Expected: {}'.format(u.shape, shouldbe_dims))
    assert most_likely_path(u, i) == [None, 'IN', 'IN', 'IN', 'DT', 'CC', 'DT', 'TO']
    
test_likely_path()


# In[ ]:

xt = 'Hi this has Two capped words'.split()


resmat = getmat(gft(0))


# In[ ]:

gft = mkgf()


# In[ ]:

def most_likely_path(u: 'DataFrame[float]', i: 'DataFrame[Y]') -> (List[str], List[float]):
    revpath = []
    revscore = []

    for c in reversed(u.columns[:]):
#         print(c)
        ix = u[c].idxmax()
        revscore.append(u[c][ix])
        revpath.append(ix)
        prevmax = i[c][ix]
#         print('ix:', ix)
#         print('prevmax:', prevmax)
        if c:
            assert u[c-1].max() == u[c-1][prevmax]
    #     break
    return revpath[::-1], revscore[::-1]


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



