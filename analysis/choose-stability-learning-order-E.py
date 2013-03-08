# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Mass Learning Experiment: Choose Diagnostic Stimuli Order

# <codecell>

import numpy as np

import model_observer as mo
import analysis_tools as at
from stats_tools import normalize

# <markdowncell>

# ## Configuration

# <codecell>

rso = np.random.RandomState(0)
stim_ratios = [0.1, 10]
cmap = at.make_cmap("lh", (0, 0, 0), (.55, .55, .55), (1, 1, 1))

# <markdowncell>

# ## Load Data
# ### Human Data

# <codecell>

# conditions and suffixes we want to load
conds = ['C-vfb-10', 'C-vfb-0.1', 'C-fb-10', 'C-fb-0.1', 'C-nfb-10']
suffixes = ['-cb0', '-cb1']

# load the data
training, posttest, experiment, queries = at.load_turk(conds, suffixes, thresh=1)

# <markdowncell>

# #### Stimuli

# <codecell>

# stimuli seen by participants
Stims = np.array([
    x.split("~")[0] for x in zip(*experiment[experiment.keys()[0]].columns)[1]])
n_trial = Stims.size
print Stims

# <markdowncell>

# ### Model Simulation Data

# <codecell>

# thresholds for what counts as 'fall'
nthresh0 = 0
nthresh = 0.4

# load the data
rawipe, ipe_samps, rawtruth, feedback, kappas = at.load_model(
    "mass-prediction-stability", nthresh0=nthresh0, nthresh=nthresh, fstim=Stims)
nofeedback = np.empty(feedback.shape[1])*np.nan
n_kappas = len(kappas)

# <markdowncell>

# #### Ratios and Kappas

# <codecell>

# round to two or one decimal places, depending on value
kappas = np.array(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# convert ratios and kappas to lists, so we can index
ratios = list(ratios)
kappas = list(kappas)

# find indices of the ratios that we want
ridx = [ratios.index(r) for r in stim_ratios]
r1 = ratios.index(1.0)

# <markdowncell>

# ## Compute diagnosticity

# <codecell>

# compute feedback likelihood for mass ratios and stimuli
model_lh = mo.IPE(
    feedback[ridx],
    ipe_samps,
    kappas,
    smooth=True)

# <markdowncell>

# Compute likelihood ratios of feedback $r_0=0.1$ and $r_0=10$

# <codecell>

# likelihood for r < 1 when r0 = 0.1
low01_lh = model_lh[0, :, :r1].sum(axis=-1)
# likelihood for r > 1 when r0 = 0.1
high01_lh = model_lh[0, :, r1+1:].sum(axis=-1)
# likelihood ratio for r0 = 0.1
lhr01 = low01_lh / high01_lh

# likelihood for r < 1 when r0 = 10
low10_lh = model_lh[1, :, :r1].sum(axis=-1)
# likelihood for r > 1 when r0 = 10
high10_lh = model_lh[1, :, r1+1:].sum(axis=-1)
# likelihood ratio for r0 = 10
lhr10 = high10_lh / low10_lh

# <markdowncell>

# We want to choose stimuli for which the likelihood ratio of feedback
# is greater than one -- this indicates that the feedback provides more
# evidence for the correct mass ratio than for the incorrect mass ratio.

# <codecell>

# find stimuli that have a high likelihood ratio under both values of
# r0 and randomly order them
match = (lhr01 >= 1.5) & (lhr10 >= 1.5)
idx0 = np.nonzero(match)[0]
rso.shuffle(idx0)
# randomly order the remaining stimuli
idx1 = np.nonzero(~match)[0]
rso.shuffle(idx1)

# concatenate the two lists of stimuli into an ordering
order = np.hstack([idx0, idx1])
print order

# <markdowncell>

# ### Plot model results of diagnostic order

# <codecell>

plt.clf()

for i, idx in enumerate(ridx):
    # compute model belief
    model_joint, model_theta = mo.ModelObserver(
	feedback[:, order][idx],
	ipe_samps[order],
	kappas,
	prior=None, p_ignore=0, smooth=True)

    # compute probability of choosing the correct ratio
    if i == 0:
	low = np.exp(model_theta[:, :r1]).sum(axis=1)
    elif i == 1:
	high = np.exp(model_theta[:, r1+1:]).sum(axis=1)

    # plot model belief
    at.plot_theta(
	2, 2, i+1,
	np.exp(model_theta),
	"fb-%s" % ratios[idx],
	exp=np.e,
	cmap=cmap,
	fontsize=14)

# plot probability of choosing correct ratio
plt.subplot(2, 2, 3)
plt.plot(low, label="r=0.1, p(r<1)")
plt.plot(high, label="r=10, p(r>1)")
plt.legend()
plt.title("Model probability of correct ratio")
plt.xlabel("Trial")

# plot likelihood ratios
plt.subplot(2, 2, 4)
plt.plot(lhr01[order], label="r=0.1, p(r<1)/p(r>1)")
plt.plot(lhr10[order], label="r=10, p(r>1)/p(r<1)")
plt.legend()
plt.title("Likelihood ratio of feedback on each trial")
plt.xlabel("Trial")

fig = plt.gcf()
fig.set_figwidth(10)
fig.set_figheight(8)

# save figure to file
at.savefig("images/model_belief_order-E.png", close=False)

# <markdowncell>

# ### Save the diagnostic ordering

# <codecell>

with open("../experiment/config/trial-order-E.txt", "w") as fh:
    for stim in Stims[order]:
        fh.write(stim + "\n")

