
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


# # Viterbi
# Example from [here](http://homepages.ulb.ac.be/~dgonze/TEACHING/viterbi.pdf)
# 
# s = StringIO()
# states2.to_csv(s)
# s.seek(0)
# sc = s.read()

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

