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
## Load stimuli
######################################################################

listpath = os.path.join(cogphysics.CPOBJ_LIST_PATH,
			"mass-towers-stability-learning~kappa-1.0")
with open(listpath, "r") as fh:
    Stims = np.array([x.split("~")[0] for x in fh.read().strip().split("\n") if x != ""])

# <codecell>

######################################################################
## Load human data
######################################################################

reload(lat)
training, posttest, experiment, queries = lat.load_turk_static(thresh=1)
conds = experiment.keys()

# <codecell>

######################################################################
## Load model data
######################################################################

reload(lat)

nthresh0 = 0
nthresh = 0.4
rawipe, ipe_samps, rawtruth, feedback, kappas = lat.process_model_turk(
    Stims, nthresh0, nthresh)
nofeedback = np.empty((feedback.shape[0], 1, 1))*np.nan

fig = plt.figure(1)
plt.clf()
lat.plot_smoothing(rawipe, Stims, 6, nthresh, kappas)
fig.set_figheight(6)
fig.set_figwidth(8)

lat.save("images/likelihood_smoothing.png", close=False)

# <codecell>

######################################################################
## Global parameters
######################################################################

n_kappas = len(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)
ratios = list(ratios)
kappas = list(kappas)

outcomes     = np.array([0, 1])                  # possible outcomes
n_trial      = Stims.size
n_outcomes   = outcomes.size                     # number of possible outcomes

f_smooth = True
p_ignore_stimulus = 0.0

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
alpha = 0.2

# <codecell>

######################################################################
## Generate fake human data
######################################################################

fig = plt.figure(2)
plt.clf()
plt.suptitle("Ideal Observer Beliefs")
cidx = 0

reload(mo)
nfake = 1000
for cond in sorted(experiment.keys()):

    group, fbtype, ratio, cb = lat.parse_condition(cond)
    if group == "MO":
	continue

    cols = experiment[cond].columns
    order = np.argsort(zip(*cols)[0])
    undo_order = np.argsort(order)

    # determine what feedback to give
    if fbtype == "nfb":
	#ridx = range(kappas.index(0.0)) + range(kappas.index(0.0)+1, n_kappas)
	#ridx = ratios.index(1.0)
	ridx = np.arange(n_kappas)
    else:
	ridx = ratios.index(ratio)
	
    fb = nofeedback[order][:, None]
    initial = np.zeros((n_kappas))
    initial[ridx] = 1
    initial = normalize(np.log(initial))[1]
    
    # learning model beliefs
    model_lh, model_joint, model_theta = mo.ModelObserver(
	ipe_samps[order], fb,
	outcomes=None,
	respond=False,
	initial=initial,
	p_ignore_stimulus=p_ignore_stimulus,
	smooth=f_smooth)

    # compute probability of falling
    p_outcomes = np.empty((n_trial,))
    for t in xrange(n_trial):
	newcond = "-".join(["MO"] + cond.split("-")[1:])
	p_outcomes[t] = np.exp(mo.predict(
	    model_theta[0, t][None],
	    outcomes[:, None], 
	    ipe_samps[order][t],
	    f_smooth))[:, 1]

    # sample responses
    responses = np.random.rand(nfake)[:, None] < p_outcomes[None]		
    experiment[newcond] = pd.DataFrame(
	responses[:, undo_order], 
	columns=cols)

    lat.plot_theta(
	1, 3, cidx+1,
	np.exp(model_theta[0]),
	cond,
	exp=1.3,
	cmap=cmap,
	fontsize=14)
    cidx += 1

conds = sorted(experiment.keys())
lat.save("images/ideal_observer_beliefs.png", close=False)

# <codecell>

reload(lat)

nboot = 1000
nsamp = 9
with_replacement = False

for cidx1, cond1 in enumerate(conds):
    arr1 = np.asarray(experiment[cond1])
    corrs = lat.bootcorr_wc(
	np.asarray(arr1), 
	nboot=nboot,
	nsamp=nsamp,
	with_replacement=with_replacement)
    meancorr = np.mean(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond1, cond1, meancorr, semcorr)

    for cidx2, cond2 in enumerate(conds[cidx1+1:]):
	arr2 = np.asarray(experiment[cond2])

	corrs = lat.bootcorr(
	    np.asarray(arr1), np.asarray(arr2), 
	    nboot=nboot, 
	    nsamp=nsamp,
	    with_replacement=with_replacement)
	meancorr = np.mean(corrs)
	semcorr = scipy.stats.sem(corrs)
	print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	    cond1, cond2, meancorr, semcorr)

    print

# <codecell>

nfell = rawipe['nfellA'] + rawipe['nfellB']

zscore = scipy.stats.zscore
#zscore = lambda a, axis, ddof: a
nanmean = scipy.stats.nanmean
nanstd = scipy.stats.nanstd

suffix = "-fb-0.1"

hdata = zscore(np.asarray(experiment['B' + suffix]), axis=1, ddof=1)
hmean = nanmean(hdata, axis=0)
hsem = nanstd(hdata, axis=0) / np.sqrt(hdata.shape[0])

sdata = zscore(ipe_samps[:, ratios.index(0.1), :, 0].T, axis=1, ddof=1)
#sdata = zscore(nfell[:, ratios.index(0.1)].T, axis=1, ddof=1)
#sdata = zscore(np.asarray(experiment['MO' + suffix]), axis=1, ddof=1)
smean = nanmean(sdata, axis=0)
ssem = nanstd(sdata, axis=0) / np.sqrt(sdata.shape[0])

print xcorr(hmean, smean)

plt.figure(3)
plt.clf()
plt.errorbar(hmean, smean, xerr=hsem, yerr=ssem, 
	     linestyle='', marker='o')
plt.xlabel("Human n=%d (no feedback)" % hdata.shape[0])
plt.ylabel("Model n=%d (uniform prior, no learning)" % sdata.shape[0])
plt.title("Human vs. model judgments")

lat.save("images/human_v_model.png", close=False)

# <codecell>

plt.figure(10)
plt.clf()

for cond in conds:
    if cond.startswith("MO"):
	continue
    if cond.endswith("nfb-10"):
	continue

    ratio = float(cond.split("-")[-1])
    arr = np.asarray(queries[cond]) == ratio
    mean = np.mean(arr, axis=0)
    sem = scipy.stats.sem(arr, axis=0, ddof=1)
    print mean
    plt.errorbar(np.arange(mean.shape[0]), mean, yerr=sem, 
		 label="r=%s (n=%d)"% (cond.split("-")[-1], arr.shape[0]))

plt.xlim(-.5, 3.5)
plt.xticks([0, 1, 2, 3], [5, 10, 15, 20])
plt.xlabel("Trial")
plt.ylabel("P(judge correct mass ratio)")
plt.title("Explicit mass judgments")
plt.legend(loc=0)

lat.save("images/explicit_mass_judgments.png", close=False)
    

# <codecell>


