
# coding: utf-8

# In[ ]:

from py3k_imports import *
import project_imports3; reload(project_imports3); from project_imports3 import *

# import warnings
# warnings.filterwarnings('ignore')

pu.psettings(pd)
pd.options.display.width = 200  # 150


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
# 

# In[ ]:

def enum(x):
    return (i for i, _ in enumerate(x[:-1], 1))


# In[ ]:

def mkfeature_deco():
    def high_feature(low):
        @wraps(low)
        def hi(xbar, ybar):
            return sum(low(ybar[i - 1], ybar[i], xbar, i) for i in enum(xbar))
        hi.low = low
        high_feature.fs.append(hi)
        return hi
    fs = high_feature.fs = []  # OrderedDict()
    return high_feature

high_feature = mkfeature_deco()

@high_feature
def is_capped(yp, y, x, i):
    return x[i][0].isupper()

@high_feature
def has_period(yp, y, x, i):
    return '.' in x[i]

@high_feature
def noun_capped(yp, y, x, i):
    return y == 'NN' and x[i][0].isupper()

# fs = [is_capped, has_period, noun_capped]
ws = np.ones_like(high_feature.fs)


# In[ ]:

x = X[0]
y = Y[0]


# In[ ]:

def mkgi(xbar, ws, i):
    def gi(ybar):
        print(i-1, i)
        return sum([f.low(ybar[i-1], ybar[i], xbar, i) * w for f, w in izip(fs, ws)])
    return gi


# In[ ]:

[mkgi(x, ws, i)(y) for i in enum(x)]


# In[ ]:

x


# In[ ]:

y


# In[ ]:

mkgi(x, ws, 1)(y)


# In[ ]:




# In[ ]:

y


# In[ ]:

high_feature.fs.values()


# ### Argmax
# $\newcommand{\argmin}{\operatornamewithlimits{argmin}}$
# Get $\text{argmax}_{\bar y} p(\bar y | \bar x;w) $
# 
# 

# In[ ]:




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



