# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# imports
import collections
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy.stats
import os
import time

import cogphysics
import cogphysics.lib.circ as circ
import cogphysics.lib.nplib as npl
import cogphysics.lib.rvs as rvs

import cogphysics.tower.analysis_tools as tat
import cogphysics.tower.mass.model_observer as mo
import cogphysics.tower.mass.learning_analysis_tools as lat

from cogphysics.lib.corr import xcorr, partialcorr

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

pd.set_option('line_width', 195)
LINE = "-"*195

# <codecell>

######################################################################
## Load human data
######################################################################

reload(lat)
training, posttest, experiment, queries = lat.load_turk_static(thresh=1)

# <codecell>

######################################################################
## Load stimuli
######################################################################

# listpath = os.path.join(cogphysics.CPOBJ_LIST_PATH,
# 			"mass-towers-stability-learning~kappa-1.0")
# with open(listpath, "r") as fh:
#     Stims = np.array(sorted([
# 	x.split("~")[0] for x in fh.read().strip().split("\n") if x != ""]))

Stims = np.array([
    x.split("~")[0] for x in zip(*experiment[experiment.keys()[0]].columns)[1]])

# <codecell>

######################################################################
## Load model data
######################################################################
reload(lat)

nthresh0 = 0
nthresh = 0.4
rawipe, ipe_samps, rawtruth, feedback, kappas = lat.process_model_turk(
    Stims, nthresh0, nthresh)
nofeedback = np.empty(feedback.shape[1])*np.nan

# <codecell>

######################################################################
## Global parameters
######################################################################

n_kappas = len(kappas)
kappas = np.array(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)
ratios = list(ratios)
kappas = list(kappas)

outcomes     = np.array([0, 1])                  # possible outcomes
n_trial      = Stims.size
n_outcomes   = outcomes.size                     # number of possible outcomes

f_smooth = True
p_ignore = 0.0

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
alpha = 0.2
colors = ['c', 'm', 'y']#'#FF9966', '#AAAA00', 'g', 'c', 'b', 'm']
#colors = cm.hsv(np.round(np.linspace(0, 220, n_cond)).astype('i8'))

# <codecell>

seed = 0
rso = np.random.RandomState(seed)

nbad = 1
n = 10

model_lh = mo.IPE(
    feedback[[ratios.index(0.1), ratios.index(10.0)]],
    ipe_samps,
    kappas,
    smooth=True)

low01_lh = model_lh[0, :, :ratios.index(1.0)].sum(axis=-1)
high01_lh = model_lh[0, :, ratios.index(1.0)+1:].sum(axis=-1)
lhr01 = low01_lh / high01_lh

low10_lh = model_lh[1, :, :ratios.index(1.0)].sum(axis=-1)
high10_lh = model_lh[1, :, ratios.index(1.0)+1:].sum(axis=-1)
lhr10 = high10_lh / low10_lh

match = (lhr01 >= 1.5) & (lhr10 >= 1.5)
#match = (lhr01 > 1) & (lhr10 > 1)
idx0 = np.nonzero(match)[0]
rso.shuffle(idx0)
idx1 = np.nonzero(~match)[0]
rso.shuffle(idx1)
order = np.hstack([idx0, idx1])
order

# <codecell>

plt.figure(6)
plt.clf()

cond = "fb-0.1"
model_joint, model_theta = mo.ModelObserver(
    feedback[:, order][ratios.index(0.1)],
    ipe_samps[order],
    kappas,
    prior=None, p_ignore=0, smooth=True)
low01 = np.exp(model_theta[:, :ratios.index(1.0)]).sum(axis=1)
high01 = np.exp(model_theta[:, ratios.index(1.0)+1:]).sum(axis=1)

lat.plot_theta(
    2, 2, 1,
    np.exp(model_theta),
    cond,
    exp=1.3,
    cmap=cmap,
    fontsize=14)


cond = "fb-10"
model_joint, model_theta = mo.ModelObserver(
    feedback[:, order][ratios.index(10.0)],
    ipe_samps[order],
    kappas,
    prior=None, p_ignore=0, smooth=True)
low10 = np.exp(model_theta[:, :ratios.index(1.0)]).sum(axis=1)
high10 = np.exp(model_theta[:, ratios.index(1.0)+1:]).sum(axis=1)

lat.plot_theta(
    2, 2, 2,
    np.exp(model_theta),
    cond,
    exp=1.3,
    cmap=cmap,
    fontsize=14)

plt.subplot(2, 2, 3)
plt.plot(low01, label="r=0.1, p(r<1)")
plt.plot(high10, label="r=10, p(r>1)")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(lhr01[order], label="r=0.1, p(r<1)/p(r>1)")
plt.plot(lhr10[order], label="r=10, p(r>1)/p(r<1)")
#plt.plot(high, label=">1")
plt.legend()

lat.save("images/model_belief_order-E.png", close=True)

# <codecell>

with open("../../turk-experiment/www/config/trial-order-E.txt", "w") as fh:
    for stim in Stims[order]:
	fh.write(stim + "\n")
    

# <codecell>


