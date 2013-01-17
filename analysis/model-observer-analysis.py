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

def plot_baserates(fignum, nsamps, smooth):
    plt.figure(fignum)
    plt.clf()
    plt.subplots_adjust(
        hspace=0.5,
        top=0.85,
        bottom=0.1,
        left=0.1,
        right=0.93,
        wspace=0.1)
    plt.suptitle(
        "Effect of model/feedback baserates on "
        "posterior over mass ratio\n(%d samples)" % (nsamps),
        fontsize=20)
    rows = np.array([0, 2, 4, 5])
    cols = np.array([0, 2, 4, 5])
    i=0
    r, c = rows.size, cols.size
    n = r*c
    for nthresh0 in rows:
        for nthresh in cols:
            feedback, ipe_samps = lat.make_observer_data(
                nthresh0, nthresh, nsamps)
            lh, jnt, th = mo.ModelObserver(
                ipe_samps,
                feedback[:, None],
                outcomes=None,
		respond=False,
		p_ignore_stimulus=0.0,
		smooth=smooth)
            plt.subplot(r, c, i+1)
            exp = np.exp(np.log(0.5) / np.log(1. / n_kappas))
            nth = normalize(th[:, -1].T, axis=0)[1]
            post = np.mean(np.exp(nth), axis=-1)
            plt.imshow(exp**nth,
                       cmap=cmap,
                       interpolation='nearest')
            plt.xlim(0, post.size-1)
            plt.ylim(0, post.size-1)
            yticks = np.round(
                np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (i%c) == 0:
                plt.yticks(yticks, ratios[yticks], fontsize=12)
                plt.ylabel("Mass ratio ($\kappa$)")
            else:
                plt.yticks(yticks, [])
                plt.ylabel("")
            xticks = np.round(
                np.linspace(0, n_kappas-1, 5)).astype('i8')
            if (n-i) <= c:
                plt.xticks(xticks, ratios[xticks], fontsize=12)
                plt.xlabel("Feedback condition ($r$)")
            else:
                plt.xticks(xticks, [])
                plt.xlabel("")
            plt.title("Model thresh=%d\n"
                      "  Fb    thresh=%d" % (
                          nthresh, nthresh0),
                      fontsize=12)
            i+=1

# <codecell>

def plot_smoothing(nstim, fignum, nthresh, nsamps):
    samps = np.concatenate([
        ((rawipe['nfellA'] + rawipe['nfellB']) > nthresh).astype(
            'int')[..., None]], axis=-1)[..., 0][..., :nsamps]
    stims = np.array([int(x.split("_")[1])
                      for x in rawhstim])
    alpha = np.sum(samps, axis=-1) + 0.5
    beta = np.sum(1-samps, axis=-1) + 0.5
    pfell_mean = alpha / (alpha + beta)
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    pfell_meanstd = np.mean(pfell_std, axis=-1)
    colors = cm.hsv(np.round(np.linspace(0, 220, nstim)).astype('i8'))
    xticks = np.linspace(-1.3, 1.3, 7)
    xticks10 = 10 ** xticks
    xticks10[xticks < 0] = np.round(xticks10[xticks < 0], decimals=2)
    xticks10[xticks >= 0] = np.round(xticks10[xticks >= 0], decimals=1)
    yticks = np.linspace(0, 1, 3)

    plt.figure(fignum)
    plt.clf()
    plt.suptitle(
        "Likelihood function for feedback given mass ratio\n"
        "(%d IPE samples, threshold=%d blocks)" % (nsamps, nthresh),
        fontsize=16)
    plt.ylim(-0.2, 1)
    plt.xticks(xticks, xticks10)
    plt.xlabel("Mass ratio ($\kappa$)", fontsize=14)
    plt.yticks(yticks, yticks)
    plt.ylabel("P(fall|$\kappa$, $S$)", fontsize=14)
    plt.grid(True)
    order = (range(0, stims.size, 2) + range(1, stims.size, 2))[:nstim]
    for idx in xrange(nstim):
        i = order[idx]
        x = kappas
        #xn = np.linspace(-1.5, 1.5, 100)
        lam = pfell_meanstd[i] * 10
        kde_smoother = mo.make_kde_smoother(x, lam)
        y_mean = kde_smoother(pfell_mean[i])
        plt.plot(x, y_mean,
                 color=colors[idx],
                 linewidth=3)        
        plt.errorbar(x, pfell_mean[i], pfell_std[i], None,
                     color=colors[idx], fmt='o',
                     markeredgecolor=colors[idx],
                     markersize=5,
                     label="Tower %d" % stims[i])
    plt.legend(loc=8, prop={'size':12}, numpoints=1,
               scatterpoints=1, ncol=3, title="Stimuli ($S$)")

# <codecell>

nthresh0 = 1
nthresh = 4
ext = ['png', 'pdf']
f_save = True
f_close = False

fbbr, ipebr = calc_baserates(nthresh0, nthresh, nsamps=300)

# <codecell>

plot_baserates(1, nsamps=48, smooth=True)
if f_save:
   lat.save("images/baserates_048samples",
	ext=ext, width=10, height=10, close=f_close)

# <codecell>

plot_baserates(2, nsamps=300, smooth=True)
if f_save:
   lat.save("images/baserates_300samples",
        ext=ext, width=10, height=10, close=f_close)

# <codecell>

lh, jnt, th = plot_belief(
    fignum=3,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=48,
    smooth=False)
if f_save:
   lat.save("images/belief_raw_048samples",
        ext=ext, width=9, height=7, close=f_close)

# <codecell>

lh, jnt, th = plot_belief(
    fignum=4,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=48,
    smooth=True)
if f_save:
   lat.save("images/belief_smoothed_048samples",
        ext=ext, width=9, height=7, close=f_close)

# <codecell>

lh, jnt, th = plot_belief(
    fignum=5,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=300,
    smooth=False)
if f_save:
   lat.save("images/belief_raw_300samples",
        ext=ext, width=9, height=7, close=f_close)

# <codecell>

lh, jnt, th = plot_belief(
    fignum=6,
    nthresh0=nthresh0,
    nthresh=nthresh,
    nsamps=300,
    smooth=True)
if f_save:
   lat.save("images/belief_smoothed_300samples",
        ext=ext, width=9, height=7, close=f_close)

# <codecell>

plot_smoothing(6, fignum=7, nthresh=nthresh, nsamps=48)
if f_save:
   lat.save("images/likelihood_smoothing_048samples",
	ext=ext, width=9, height=7, close=f_close)

# <codecell>

plot_smoothing(6, fignum=8, nthresh=nthresh, nsamps=300)
if f_save:
   lat.save("images/likelihood_smoothing_300samples",
        ext=ext, width=9, height=7, close=f_close)

# <codecell>


