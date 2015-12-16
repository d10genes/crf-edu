
# coding: utf-8

# # Linear chain CRFs
# This is less of a blog post, and more of my annotated progress in implementing CRF's based on Charles Elkan's very excellent [video](http://videolectures.net/cikm08_elkan_llmacrf/) and [pdf](http://cseweb.ucsd.edu/~elkan/250Bwinter2012/loglinearCRFs.pdf) tutorials in order to have a better understanding of log-linear models.

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

import utils; reload(utils); from utils import *
import crf; reload(crf); from crf import *
FeatUtils.bookend = False


# In[ ]:

Series.__matmul__ = Series.dot
DataFrame.__matmul__ = DataFrame.dot

from matmul_new import test_matmul
test_matmul()


# ## Probabilistic model
# 
# Given a sequence $\bar x$, the linear chain CRF model gives the probability of a corresponding sequence $\bar y$ as follows, for feature functions $F_j$, where each $F_j$ is a sum of a corresponding lower level feature function $f_j$ over every element of the sequence:
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
# $Z(\bar x, w)$ is the partition function, that sums the probabilities of all possible sequences to normalize the probability:
# 
# $$
# Z(x, w) = \sum_{y' \in Y} \exp \sum_{j=1} ^J w_j F_j (x, y').
# $$

# ### Argmax
# 
# Computing the most likely sequence $\text{argmax}_{\bar y} p(\bar y | \bar x;w)$ naively involves iterating over every possible sequence that can be built from the tag vocabulary, rendering the computation impractical for even medium sized tag-spaces.
# 
# Since the scoring function only depends on 2 (consecutive in this situation) elements of $\bar y$, argmax can be computed in polynomial time with a table ($\in ℝ^{|Y| \times |y|}$). $U_{ij}$ is the highest score for sequences ending in $y_i$ at position $y_j$. It is useful to compute the most likely sequence in terms of $g_i$, which sums over all lower level functions $f_j$ evaluated at position $i$:

# $$
# g_i(y_ {i-1}, y_i) = \sum^J_{j=1} w_j f_j (y_ {i-1}, y_i, \bar x, i)
# $$

# ### Generate maximum score matrix U
# 
# $$U(k, v) = \max_u [U(k-1, u) + g_k(u,v)]$$
# $$U(1, vec) = \max_{y_0} [U(0, y_0) + g_k(y_0,vec)]$$
# 
# This implementation is pretty slow, because every low level feature function is evaluated at each $i, y_{i-1}$ and $y_i$, for each feature function $f_j$ ($\mathcal{O}(m^2 n J )$ where $J=$ number of feature functions, $m=$ number of possible tags and $n=$ length of sequence $\bar y$). Also, using python functions in the inner-loop is slow. This could be significantly reduced if the feature functions could be arranged such that they would only be evaluated for the relevant combinations of $x_i, y_{i-1}$ and $y_i$. I started arranging them in this way in `dependency.py`, but the complexity got a bit too unwieldy for a toy educational project. 

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


# In[ ]:

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

def predict(xbar=None, fs=None, tags=None, ws=None, gf=None):
    "Return argmax_y with corresponding score"
    if gf is None:
        ws = ws or mkwts1(fs)
        gf = G(ws=ws, fs=fs, tags=tags, xbar=xbar)
    u, i = get_u(gf=gf, collect=True, verbose=0)
    path = mlp(i)
    return path, u.ix[END].iloc[-1]
    
# path2, score2 = predict(xbar=EasyList(['wd1', 'pre-end', 'whatevs']),
#                         fs=no_test_getu3.fs,
#                         tags=[START, 'TAG1', 'PENULTAG', END])


# In[ ]:

import test; reload(test); from test import *
 
no_test_getu1(get_u, mlp)
no_test_getu2(get_u, mlp)
no_test_getu3(get_u, mlp)

test_corp()


# ##Gradient
# $$\frac{\partial}{\partial w_j} \log p(y | x;w) = F_j (x, y) - \frac1 {Z(x, w)} \sum_{y'} F_j (x, y') [\exp \sum_{j'} w_{j'} F_{j'} (x, y')]$$
# $$= F_j (x, y) - E_{y' \sim  p(y | x;w) } [F_j(x,y')]$$
# 
# 
# ### Forward-backward algorithm
# - Partition function $Z(\bar x, w) = \sum_{\bar y} \exp \sum _{j=1} ^ J w_j F_j (\bar x, \bar y) $ can be intractible if calculated naively (similar to argmax); forward-backward vectors can make it easier to compute
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

# Check correctness of forward and backward vectors.
# - $ Z(\bar x, w) = \beta(START, 0) = \alpha(n+1, END) $
# - For all positions $k=0...n+1$, $\sum_u \alpha(k, u) \beta(u, k) = Z(\bar x, w)$

# In[ ]:

def mk_asum(gf, vb=False):
    n = len(gf.xbar)
    tags = gf.tags
    p = testprint(vb)
    
    @memoize
    def get_asum(knext=None):
        if knext is None:
            # The first use of the forward vectors is to write
            return get_asum(n+1)
        if knext < 0:
            raise ValueError('k ({}) cannot be negative'.format(k))
        if knext == 0:
            return init_score(tags, tag=START)
        k = knext - 1
        gnext = gf(knext).mat
        ak = get_asum(k)

        if vb:
            names = 'exp[g{k1}] g{k1} a_{k}'.format(k1=knext, k=k).split()
            p(side_by_side(np.exp(gnext), gnext, ak, names=names))
        # expsum = Series([sum([ak[u] * np.exp(gnext.loc[u, v]) for u in tags]) for v in tags], index=tags)
        # vectorizing is much faster:
        expsum = np.exp(gnext).mul(ak, axis=0).sum(axis=0)
        return expsum
    return get_asum  #(knext, vb=vb)


def mk_bsum(gf, vb=False):
    p = testprint(vb)
    n = len(gf.xbar)
    tags = gf.tags
    
    @memoize
    def get_bsum(k=None):
        if k is None:
            return get_bsum(0)
        if k > n + 1:
            raise ValueError('{} > length of x {} + 1'.format(k, n))
        if k == n + 1:
            return init_score(gf.tags, tag=END)
        gnext = gf(k + 1).mat
        bnext = get_bsum(k + 1)
        if vb:
            names = ['exp[g{}]'.format(k+1), 'g{}'.format(k+1), 'b_{}'.format(k+1)]
            p(side_by_side(np.exp(gnext), gnext, bnext, names=names))
        # expsum = Series([sum([np.exp(gnext.loc[u, v]) * bnext[v] for v in tags]) for u in tags], index=tags)
        expsum = np.exp(gnext).mul(bnext, axis=1).sum(axis=1)
        return expsum
    return get_bsum


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

    amkr = mk_asum(gf)
    bmkr = mk_bsum(gf)
    za = amkr().END
    zb = bmkr().START
    assert za == zb
    
    for k in range(len(x) + 2):
        assert amkr(k) @ bmkr(k) == za
    return za
    
test_fwd_bkwd()


# ### Calculate expected value of feature function
# Weighted by conditional probability of $y'$ given $x$
# 
# $$
# E_{\bar y \sim  p(\bar y | \bar x;w) } [F_j(\bar  x, \bar y)] =
# \sum _{i=1} ^n \sum _{y_{i-1}} \sum _{y_i}
#     f_j(y_{i-1}, y_i, \bar x, i)
#     \frac {\alpha (i-1, y_{i-1})
#     [\exp g_i(y_{i-1}, y_i)]
#     \beta(y_i, i)
#     }
#     {Z(\bar x, w)}
# $$

# In[ ]:

def sdot(s1: Series, s2: Series):
    d1, d2 = s1.values[:, None], s2.values[:, None]
    return d1 @ d2.T
#     return DataFrame(d1 @ d2.T, columns=s1.index, index=s2.index)


# In[ ]:

def expectation2(gf, fj):
    "Faster matrix multiplication version"
    tags = gf.tags
    n = len(gf.xbar)
    ss = 0
    ss2 = 0
    asummer = mk_asum(gf)
    bsummer = mk_bsum(gf)
    
    za = partition(asummer=asummer)
    global α, β, alpha_vec, beta_vec, gfix, smat
    
    def sumi(i):
        gfix = np.exp(gf(i).mat.values)
        alpha_vec = asummer(i - 1)
        beta_vec = bsummer(i)
        fmat = np.array([[fj(yprev, y, gf.xbar, i) for y in tags] for yprev in tags])
        smat = sdot(alpha_vec, beta_vec) * gfix * fmat
        return smat.sum() #.sum()
    
    return sum([sumi(i) for i in range(1, n + 2)]) / za

def expectation_(gf, fj):
    "Slow, looping version"
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


#     %time expectation2(gf, gfsub.fs['ly_VBZ'])
#     %prun -qD profexp.prof expectation2(gf, gfsub.fs['ly_VBZ'])
# 
#     %prun -qD profexp.prof train(Zstrn, gfsub, maxiter=1, tol=.005)

# ##Partial derivative
# ###Probability function

# In[ ]:

def partial_d(gf, fj, y, Fj=None) -> float:
    f = fj if callable(fj) else gf.fs[fj]
    if Fj is None:
        Fj = FeatUtils.mk_sum(f)
    #ex1 = expectation(gf, f)
    ex2 = expectation2(gf, f)
    #assert np.allclose(ex1, ex2)
    return Fj(gf.xbar, y) - ex2


def prob(gf, y, norm=True):
    Fs = z.valmap(FeatUtils.mk_sum, gf.fs)
    p = np.exp(sum([Fj(gf.xbar, y) * gf.ws[fname] for fname, Fj in Fs.items()]))
    if not norm:
        return p
    za = partition(gf=gf)
    return p / za
     

def partition(gf=None, asummer=None):
    assert asummer or gf, 'Supply at least one argument'
    asummer = asummer or mk_asum(gf)
    return asummer().END


#     import autograd.numpy as np
#     def probf(f, w):
# 

#     %time expectation(gf, fj)
#     %time expectation_(gf, fj)

#     %prun -qD profex.prof expectation2(gf, fj)

# %time expectation(gf, fj)

# In[ ]:

corpus = '''Nothing seems hard here .//NN VBZ JJ RB .
The reason is cost .//DT NN VBZ NN .
Terms were n't disclosed .//NNS VBD RB VBN .
Mr. Juliano really really thinks so .//NNP NNP RB RB VBZ RB .
Mr. Bill seems dead .//NNP NNP VBZ JJ .
Young & Rubicam 's Pact//NNP CC NNP POS NNP
Albany escaped embarrassingly unscathed .//NNP VBD RB JJ .'''

corp3 = '''Nothing seems hard//NN VBZ JJ
The reason is//VBZ NN VBZ
Terms were n't//UNK UNK RB
Mr. Juliano really really thinks//NNP NNP RB
Mr. Bill seems//NNP NNP VBZ
Young & Rubicam//NNP VBZ NNP
Albany escaped embarrassingly//NNP UNK RB'''

def mk_fx_tag(fx, tag):
    def f(yp_, y, x, i):
        return x[i] and fx(x[i]) and (y == tag)
    f.__name__ = '{}(x)_{}'.format(fx, tag)
    f.__doc__ = '{}(x[i]) and (y == {})'.format(fx, tag)
    return f


def mkgf(x=None, corpus=corpus):
    xs, ys = process_corpus(corpus)
    zs = zip(xs, ys)
    tgs = sorted({y for ybar in ys for y in ybar.aug})

    iscapped = lambda x: x and x[0].isupper()
    fs = dict(
        seems_VBZ=mk_word_tag('seems', 'VBZ'),
        ly_VBZ=lambda yp, y, x, i: x[i] and x[i].endswith('ly') and (y == 'RB'),
        cap_NN=mk_fx_tag(iscapped, 'NN'),
        cap_NNP=mk_fx_tag(iscapped, 'NNP'),
        nocap_START=lambda yp, y, x, i: x[i] and not iscapped(x[i]) and (yp == START),
    #     cap_NN=lambda yp, y, x, i: iscapped(x[i]) and (y == 'NN'),
    )
    return G(fs=fs, tags=tgs, xbar=x or xs[-1], ws=mkwts1(fs, 1)), ys, zs

def test_corp():
    gf, _, zs = mkgf(x=None, corpus=corpus)
    Fs = z.valmap(FeatUtils.mk_sum, gf.fs)
    
    def runFs(Fj, zs=zs):
        return [Fj(x, y) for x, y in zs]
    
    assert sum(runFs(Fs['ly_VBZ'])) == 3
    assert sum(runFs(Fs['cap_NNP'])) == 8
    assert sum(runFs(Fs['cap_NN'])) == 1
    assert not sum(runFs(Fs['nocap_START']))


# ## Test Partial

#     Fs = z.valmap(FeatUtils.mk_sum, gf.fs)
#     Fs
#     Fj = Fs['ly_VBZ']
#     Fj(gf.xbar, ybar)
#     ybase = ys3[-1].aug[1:-1]
#     ybase
#     yars = [[t] + ybase for t in gf.tags]
#     tgs = sorted(set(gf.tags) - {START, END})

# ## All length-3 sequences

#     gf, ys3, zs3 = mkgf(corpus=corp3)
#     fj = gf.fs['ly_VBZ']
#     # partial_d(gf, fj, ys[-1], Fj=None)
#     Y3 = [AugmentY([y1, y2, y3]) for y1 in tgs for y2 in tgs for y3 in tgs ]
# 
# 
#     # list(enumerate(Y3))
#     ybar = Y3[105]
#     ybar
#     prob(gf, ybar, norm=False)
# 
# 
#     Y3[:2]
# 
# 
#     asummer = mk_asum(gf)
#     side_by_side(asummer(0), asummer(1), asummer(2), asummer(3), asummer(4), )
# 
#     def gcalc(gf, ybar):
#         return np.exp(sum([gf(i)(yp, y) for i, yp, y in zip(count(1), ybar.aug, ybar.aug[1:-1])]))
# 
#     gpart = sum(gcalc(gf, y) for y in Y3)
#     ps = sum([prob(gf, y, norm=False) for y in Y3])
#     assert gpart == ps
# 
#     side_by_side(gf(0).mat, np.exp(gf(0).mat), )
# 
#     nudge = lambda x, eps=.001: x + eps
#     p1 = lambda x: x + 1
#     bump = z.partial(nudge, eps=-.001)
# 
#     zs = zip(*process_corpus(corpus))
#     for xi, yi in zs:
#         gf, _ = mkgf(x=xi)
#         for j in gf.fs:
#     #         print(j)
#             ws2 = z.update_in(gf.ws, [j], bump)
#             gf2 = gf._replace(ws=ws2)
#             break
# 
#     print('ly in gf.xbar?:', any(map(lambda x: x.endswith('ly'), gf.xbar)))
#     j
# 
#     gf.diff(gf2)

# ### Train

# gf = G(fs=fs, tags=tgs, xbar=x, ws=ws)

# In[ ]:

gf, ys, zs = mkgf(corpus=corpus)
fj = gf.fs['ly_VBZ']


# In[ ]:

λ = 1


def train_(zs: List[Tuple[EasyList, AugmentY]],
          fjid='ly_VBZ', fs=None, ws=None, vb=True, tgs=None):
    fj = fs[fjid]
    Fj = FeatUtils.mk_sum(fj)
    pt = testprint(vb)
    for x, y in zs:
        gf_ = G(fs=fs, tags=tgs, xbar=x, ws=ws)
        if not Fj(x, y):  # TODO: is this always right?
            continue
#         print(gf)
#         print(gf_)
        pder = partial_d(gf_, fj, y, Fj=Fj)
        wj0 = ws[fjid]
        ws[fjid] += λ * pder
        pt('wj: {} -> {}'.format(wj0, ws[fjid]))
        pt('pder: {:.2f}'.format(pder), Fj(x, y))
    return ws

def train_j(zs: List[Tuple[EasyList, AugmentY]],
          fjid='ly_VBZ', fs=None, ws=None, tol=.01, maxiter=10, vb=True, tgs=None):
    ws1 = ws
    pt = testprint(vb)
    
    for i in count(1):
        pt('Iter', i)
        wj1 = ws1[fjid]
        ws2 = train_(zs, fjid=fjid, fs=fs, ws=ws1, vb=vb, tgs=tgs)
        wj2 = ws2[fjid]
        if abs((wj2 - wj1) / wj1) < tol or (i >= maxiter):
            return ws, i
        ws1 = ws2
        
def train(zs, gf, ws=None, tol=.001, maxiter=10, vb=False):
    wst = (ws or gf.ws).copy()
    for fname, f in gf.fs.items():
        wst, i = train_j(zs, fjid=fname, fs=gf.fs, ws=wst, tol=tol, maxiter=maxiter, vb=vb, tgs=gf.tags)
        print(fname, 'trained in', i, 'iters: {:.2f}'.format(wst[fname]))
        sys.stdout.flush()
    return wst

# %time ws1c = train(zs, gf, mkwts1(gf.fs, 1), maxiter=100, tol=.005)


# ## Load data
# 
# with open('data/pos.train.txt','r') as f:
#     txt = f.read() #
# sents = filter(None, [zip(*[e.split() for e in sent.splitlines()]) for sent in txt[:].split('\n\n')])
# X = map(itg(0), sents)
# Y_ = map(itg(1), sents)
# Xa = map(EasyList, X)
# Ya = map(AugmentY, Y_)
# tags = sorted({tag for y in Y_ for tag in y if tag.isalpha()})
# txt[:100]
# # common bigrams
# bigs = defaultdict(lambda: defaultdict(int))
# 
# for y in Y_:
#     for t1, t2 in zip(y[:-1], y[1:]):
#         bigs[t1][t2] += 1
#         
# bigd = DataFrame(bigs).fillna(0)[tags].ix[tags]
# # bigd
# # sns.clustermap(bigd, annot=1, figsize=(16, 20), fmt='.0f')
# wcts_all = defaultdict(Counter)
# for xi, yi in zip(X, Y_):
#     for xw, yw in zip(xi, yi):
#         wcts_all[xw][yw] += 1
# wcts = z.valfilter(lambda x: sum(x.values()) > 4, wcts_all)
# ' '.join(y)
# ' '.join(tags)

# ## Evaluation
# Since I'm maximizing the log-likelihood during testing, that would seem a natural measure to evaluate improvement. I'm a bit suspicious about bugs in my implementation, so I'd like to evaluate Hamming distance to see how much the predictions improve.   

# In[ ]:

def hamming(y, ypred, norm=True):
    sm = sum(a != b for a, b in zip(y, ypred))
    return sm / len(y) if norm else sm


# In[ ]:

Zs = zip(Xa, Ya)
Zstrn = Zs[:1]
Ztst = Zs[50:]


# In[ ]:

for x, y in Ztst:
    break
x


# In[ ]:

get_ipython().magic('timeit predict(gf=gg)')


# In[ ]:

get_ipython().magic('prun -qD pred.prof predict(gf=gg)')


# In[ ]:

import utils; reload(utils); from utils import *
import crf; reload(crf); from crf import *


# In[ ]:

gg = G(fs=crf.fs, tags=sorted([START, END] + tags), xbar=EasyList(x), ws=rand_weights(crf.fs))

gg.Gi


# In[ ]:

- profile predict
    - predict will be fundamentally slow due to the naive use of feature functions


# In[ ]:

gg = G(fs=crf.fs, tags=sorted([START, END] + tags), xbar=EasyList(x), ws=rand_weights(crf.fs))

def ev(zs, score=hamming):
    def eval(x, y):
        g = gg._replace(xbar=EasyList(x))
        ypred = predict(gf=g)[0]
        return score(y.aug, ypred)
    
    return Series([eval(x, y) for x, y in zs])


# In[ ]:

scores = ev(Ztst[:5])


# In[ ]:

scores


# In[ ]:

scores.mean()


# In[ ]:

hamming(y.aug, ypred, 1)


# In[ ]:

y.aug


# In[ ]:

ypred, _ = predict(gf=gg)


# In[ ]:

y


# In[ ]:

gf.


# In[ ]:

tags


# In[ ]:




# In[ ]:

gfsub = gf._replace(fs=dict(ly_VBZ=gf.fs['ly_VBZ']))
gfsub = gfsub._replace(ws=rand_weights(gfsub.fs), tags=sorted(tags + [START, END]))


# In[ ]:

get_ipython().magic('prun -qD proftrn.prof train(Zstrn, gfsub, maxiter=1, tol=.005)')


# In[ ]:

get_ipython().magic('prun -qD proftrn.prof train(Zstrn, gf._replace(tags=sorted(tags + [START, END]), ws=rand_weights(gf.fs)), maxiter=1, tol=.005)')


# In[ ]:

get_ipython().magic('time wst2b = train(Zstrn, gf._replace(tags=sorted(tags + [START, END]), ws=rand_weights(gf.fs)), maxiter=5, tol=.005)')


# In[ ]:

get_ipython().magic('time wst2 = train(Zstrn, gf._replace(tags=sorted(tags + [START, END]), ws=rand_weights(gf.fs)), maxiter=5, tol=.005)')


# In[ ]:

# %time wst = train(Zstrn, gf._replace(tags=sorted(tags + [START, END])), mkwts1(gf.fs, 1), maxiter=100, tol=.005)


# In[ ]:

get_ipython().magic('time wst2 = train(Zstrn, gf._replace(tags=sorted(tags + [START, END]), ws=rand_weights(gf.fs)), maxiter=5, tol=.005)')


# In[ ]:

Ya[:2]


# In[ ]:

gf_ = gf._replace(ws=wst2)
for x, y in Ztst[:5]:
    gfx = gf_._replace(xbar=x)
    ypred, sc = predict(gf=gfx)
    print(DataFrame(zip(gfx.xbar, ypred, y.aug)))
#     break


# In[ ]:




# In[ ]:

predict()


# wst == {'cap_NN': 5.5770337170150999,
#  'cap_NNP': 6.5795482596835342,
#  'ly_VBZ': 7.1610907498951821,
#  'nocap_START': -4.0968133029466216,
#  'seems_VBZ': 1}
#  
#  wst2 == {'cap_NN': 5.6535298922935278,
#  'cap_NNP': 6.6593411043988899,
#  'ly_VBZ': 7.2107462564832634,
#  'nocap_START': -4.0972608563379938,
#  'seems_VBZ': 1.8675579901499675}

# In[ ]:

def likelihood(gf, zs, ws=None):
    if ws:
        gf = gf._replace(ws=ws)
    return sum([prob(gf._replace(xbar=x), y, norm=False) for x, y in zs]) / (len(zs) * partition(gf))
    for x, y in zs:
        gfx = gf._replace(xbar=x, ws=ws1c)
        print(prob(gfx, y, norm=False))


# In[ ]:

likelihood(gf, Ztst, ws=None)


# In[ ]:

rand_weights(gf.fs)


# In[ ]:

partition(gf)


# In[ ]:

likelihood(gf, Ztst, ws=ws1c)


# In[ ]:

Xa
Ya
tags


# In[ ]:

for x, y in zs:
    gfx = gf._replace(xbar=x)
    print(prob(gfx, y, norm=False))


# In[ ]:

for x, y in zs:
    gfx = gf._replace(xbar=x, ws=ws1c)
    print(prob(gfx, y, norm=False))


# In[ ]:

prob(gf, )


# In[ ]:

ws1c


# In[ ]:

train()


# In[ ]:

train_j(zs, fjid='ly_VBZ', ws=ws)


# In[ ]:

get_ipython().magic('time ws1 = train(zs, fs, mkwts1(fs, 1), maxiter=1)')


# In[ ]:

ws1c


# In[ ]:

ws1


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


# ## Extra
