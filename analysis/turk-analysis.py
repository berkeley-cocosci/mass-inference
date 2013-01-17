# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# imports
import collections
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
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

from cogphysics.lib.corr import xcorr

# <codecell>

# global variables
normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))

# <codecell>

######################################################################
## Load and process old stability data
out = lat.load('stability')
rawhuman0, rawhstim0, raworder0, rawtruth0, rawipe0, kappas = out

# <codecell>

######################################################################
## Load and process new data
hdata = np.load("../../turk-experiment/data.npz")
rawhuman = hdata['data']['response'][..., None]
rawhstim = np.array([x.split("~")[0] for x in hdata['stims']])
raworder = hdata['data']['trial'][..., None]

idx = np.nonzero((rawhstim0[:, None] == rawhstim[None, :]))[0]
rawtruth = rawtruth0[idx].copy()
rawipe = rawipe0[idx].copy()

# <codecell>

ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# XXX: when you get more data you need to change this to actually
# order for each participant! currently it only orders by the first
# one!
human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, rawtruth, rawipe)
truth = truth[0]
ipe = ipe[0]
 
# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)

# <codecell>

def calc_baserates(nthresh0, nthresh, nsamps):
    feedback, ipe_samps = lat.make_observer_data(
        nthresh0, nthresh, nsamps)

    fbbr = np.mean(np.swapaxes(feedback, 0, 1).reshape((-1, n_kappas)))
    ipebr = np.mean(np.swapaxes(ipe_samps[..., 0], 0, 1).reshape((-1, n_kappas)))

    return fbbr, ipebr

# <codecell>

def plot_belief(fignum, nthresh0, nthresh, nsamps, smooth):
    feedback, ipe_samps = lat.make_observer_data(
        nthresh0, nthresh, nsamps)
    model_lh, model_joint, model_theta = mo.ModelObserver(
        ipe_samps,
        feedback[:, None],
        outcomes=None,
	respond=False,
	p_ignore_stimulus=0.0,
        smooth=smooth)
    r, c = 3, 3
    n = r*c
    exp = np.exp(np.log(0.5) / np.log(1./27))    
    fig = plt.figure(fignum)
    plt.clf()
    gs = gridspec.GridSpec(r, c+1, width_ratios=[1]*c + [0.1])
    plt.suptitle(
        "Posterior belief about mass ratio over time\n"
        "(%d samples, %s, model thresh=%d blocks, "
        "fb thresh=%d blocks)" % (
            nsamps, "smoothed" if smooth else "unsmoothed",
            nthresh, nthresh0),
        fontsize=16)
    plt.subplots_adjust(
        wspace=0.2,
        hspace=0.3,
        left=0.1,
        right=0.93,
        top=0.85,
        bottom=0.1)
    kidxs = [0, 3, 6, 10, 13, 16, 20, 23, 26]
    for i, kidx in enumerate(kidxs):
        irow, icol = np.unravel_index(i, (r, c))
        ax = plt.subplot(gs[irow, icol])
        kappa = kappas[kidx]
        subjname = "True $\kappa=%s$" % float(ratios[kidx])
        img = lat.plot_theta(
            None, None, ax,
            np.exp(model_theta[kidx]),
            subjname,
            exp=exp,
            cmap=cmap,
            fontsize=14)
        yticks = np.round(
            np.linspace(0, n_kappas-1, 5)).astype('i8')
        if (i%c) == 0:
            plt.yticks(yticks, ratios[yticks], fontsize=14)
            plt.ylabel("Mass ratio ($\kappa$)", fontsize=14)
        else:
            plt.yticks(yticks, [])
            plt.ylabel("")
        xticks = np.linspace(0, n_trial, 4).astype('i8')
        if (n-i) <= c:
            plt.xticks(xticks, xticks, fontsize=14)
            plt.xlabel("Trial number ($t$)", fontsize=14)
        else:
            plt.xticks(xticks, [])
    logcticks = np.array([0, 0.001, 0.05, 0.25, 1])
    cticks = np.exp(np.log(logcticks) * np.log(exp))
    cax = plt.subplot(gs[:, -1])
    cb = fig.colorbar(img, ax=ax, cax=cax, ticks=cticks)
    cb.set_ticklabels(logcticks)
    cax.set_title("$P_t(\kappa)$", fontsize=14)
    return model_lh, model_joint, model_theta

# <codecell>

nthresh0 = 1
nthresh = 4
nsamps = 300
ext = ['png', 'pdf']
f_save = True
f_close = False

fbbr, ipebr = calc_baserates(nthresh0, nthresh, nsamps=300)

# <codecell>

feedback, ipe_samps = lat.make_observer_data(
    nthresh0, nthresh, nsamps)
model_lh, model_joint, model_theta = mo.ModelObserver(
    ipe_samps,
    feedback[:, None],
    outcomes=None,
    respond=False,
    p_ignore_stimulus=0.0,
    smooth=True)

# <codecell>

# global parameters
outcomes     = np.array([0, 1])                  # possible outcomes
responses    = np.array([0, 1])                  # possible responses
n_trial      = stimuli.shape[1]                  # number of trials
n_kappas     = len(kappas)                       # number of mass ratios to consider
n_responses  = responses.size                    # number of possible responses
n_outcomes   = outcomes.size                     # number of possible outcomes
kappa0       = 1.0                               # value of the true log mass ratio
ikappa0      = np.nonzero(kappas==1.0)[0][0]     # index for the true mass ratio

f_smooth = True
p_ignore_stimulus = 0.0

# <codecell>

ll = np.zeros((n_kappas, n_trial))

for sidx in xrange(n_kappas):
    
    for t in xrange(0, n_trial):

	thetas_t = model_theta[sidx, t][None]
        samps_t = ipe_samps[t]
	
	# compute likelihood of outcomes
	p_outcomes = np.exp(mo.predict(
	    thetas_t, outcomes[:, None], samps_t, f_smooth))
	
	# observe response
	response_t = int(human[0, t])
	
	# compute likelihood of response
	p_response = (p_outcomes[:, response_t]*(1-p_ignore_stimulus) + 
		      (1./n_responses)*(p_ignore_stimulus))

	ll[sidx, t] = np.log(p_response)

# <codecell>

plt.plot(np.sum(ll, axis=1))

