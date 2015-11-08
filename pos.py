
# coding: utf-8

# In[ ]:

from py3k_imports import * 
import project_imports3; reload(project_imports3); from project_imports3 import *

import warnings
warnings.filterwarnings('ignore')

pu.psettings(pd)
pd.options.display.width = 200  # 150
get_ipython().magic('matplotlib inline')


# In[ ]:

get_ipython().run_cell_magic('javascript', '', "IPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-k','ipython.move-selected-cell-up')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Ctrl-j','ipython.move-selected-cell-down')\nIPython.keyboard_manager.command_shortcuts.add_shortcut('Shift-m','ipython.merge-selected-cell-with-cell-after')")


# In[ ]:

from collections import defaultdict
import inspect
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
Y = map(itg(1), sents)
tags = sorted({tag for y in Y for tag in y if tag.isalpha()})


# In[ ]:

txt[:100]


# In[ ]:

# common bigrams
bigs = defaultdict(lambda: defaultdict(int))

for y in Y:
    for t1, t2 in zip(y[:-1], y[1:]):
        bigs[t1][t2] += 1
        
bigd = DataFrame(bigs).fillna(0)[tags].ix[tags]
# bigd
# sns.clustermap(bigd, annot=1, figsize=(16, 20), fmt='.0f')


# In[ ]:

from collections import Counter


# In[ ]:

wcts_all = defaultdict(Counter)
for xi, yi in zip(X, Y):
    for xw, yw in zip(xi, yi):
        wcts_all[xw][yw] += 1


# In[ ]:

wcts = z.valfilter(lambda x: sum(x.values()) > 4, wcts_all)


# In[ ]:

wcts


# In[ ]:

cts = Series(z.valmap(lambda x: sum(x.values()), wcts))
cts.value_counts(normalize=1)


# In[ ]:

wcts


# In[ ]:

c = Counter()
c[1] += 1
c


# In[ ]:




# In[ ]:

y


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

import utils; reload(utils)
from utils import *
# from utils import sum1, sum2, post_mr, mk_sum, F


# In[ ]:

def eq(x):
    return lambda y: x == y

def sch(term, x=False):
    f = eq(term) if isinstance(term, (str, unicode)) else term
    ss = X if x else Y
    for i, s in enumerate(ss):
        if any(f(t) for t in s):
            yield X[i], Y[i]


# In[ ]:

g = sch('Mr.', 1)


# In[ ]:

x, y = g.__next__()


# In[ ]:

x = X[0]
y = Y[0]


# ### Argmax
# $\newcommand{\argmin}{\operatornamewithlimits{argmin}}$
# Get $\text{argmax}_{\bar y} p(\bar y | \bar x;w) $
# 
# 

# In[ ]:

fs


# In[ ]:

ws = np.ones_like(fs.values())
ws = z.valmap(const(1), fs)
ws


# $$
# g_i(y_ {i-1}, y_i) = \sum^J_{j=1} w_j f_j (y_ {i-1}, y_i, \bar x, i)
# $$

# In[ ]:

def test_mats():
    xt = 'Hi this has Two capped words'.split()
    testfs = z.keyfilter(lambda x: x == 'cap_nnp', fs)
    stags = ['NNP', 'DT', 'IN', 'DERP']
    wst = z.valmap(const(1), fs)

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


# In[ ]:

gf0 = gft(0)
gf1 = gft(1)

m0 = getmat(gft(0))
m1 = getmat(gft(1))


# In[ ]:

m1


# In[ ]:

m0


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
    return gf

def getmat(gf):
    df = DataFrame({ytag: {ytag_prev: gf(ytag_prev, ytag) for ytag_prev in gf.tags}
                    for ytag in gf.tags})
    df.columns.name, df.index.name = 'Y', 'Yprev'
    return df
gf = mkgf(ws, fs, tags, ['Mr.', 'Happy', 'derp'])


# In[ ]:

gf1 = gf(1)
gf1


# In[ ]:

getmat(gf1)


# In[ ]:

x


# In[ ]:

gf1()


# In[ ]:

F.feats.values()


# In[ ]:

gf1 = gf(0)
for ytag1 in tags:
    for ytag2 in tags:

        1
ytag1


# In[ ]:




# In[ ]:

f = fs['cap_nnp']
f(None, 'NNP', ['Ca', 'lower'], 0)


# In[ ]:

d = getmat(gf(0))
d


# In[ ]:

d = getmat(gf(1))
d


# In[ ]:

getmat(gf(0))


# In[ ]:

x


# In[ ]:




# In[ ]:

DataFrame({ytag2: {ytag1: gf1(ytag1, ytag2) for ytag1 in tags}
    for ytag2 in tags})


# In[ ]:




# In[ ]:

gf(ytag1, ytag2, 1)


# In[ ]:

f()


# In[ ]:

[ytag for ytag in tags]


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



