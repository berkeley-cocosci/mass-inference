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
training, posttest, experiment, queries = lat.load_turk(thresh=1)
conds = experiment.keys()
Stims = np.array([x.split("~")[0] for x in zip(*experiment[experiment.keys()[0]].columns)[1]])

# <codecell>

######################################################################
## Load model data
######################################################################

reload(lat)

nthresh0 = 0
nthresh = 0.4
rawipe, ipe_samps, rawtruth, feedback, kappas = lat.process_model_turk(
    Stims, nthresh0, nthresh)
nofeedback = np.empty((feedback.shape[1]))*np.nan

fig = plt.figure(1)
plt.clf()
lat.plot_smoothing(ipe_samps, Stims, 6, kappas)
fig.set_figheight(6)
fig.set_figwidth(8)

lat.save("images/likelihood_smoothing.png", close=True)

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
p_ignore = 0.0

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
alpha = 0.2

# <codecell>

######################################################################
## Generate fake human data
######################################################################

# fig = plt.figure(2)
# plt.clf()
# plt.suptitle("Ideal Observer Beliefs")
# cidx = 0

reload(mo)
nfake = 2000
for cond in sorted(experiment.keys()):

    group, fbtype, ratio, cb = lat.parse_condition(cond)
    if group in ("MO", "old"):
	continue
    if fbtype == "vfb":
	fbtype == "fb"

    cols = experiment[cond].columns
    order = np.argsort(cols)
    undo_order = np.argsort(order)

    # determine what feedback to give
    if fbtype == "nfb":
	#ridx = range(kappas.index(0.0)) + range(kappas.index(0.0)+1, n_kappas)
	#ridx = ratios.index(1.0)
	ridx = np.arange(n_kappas)
    else:
	ridx = ratios.index(ratio)
	
    fb = nofeedback[..., order]
    prior = np.zeros((n_kappas))
    prior[ridx] = 1
    prior = normalize(np.log(prior))[1]
    
    # learning model beliefs
    model_joint, model_theta = mo.ModelObserver(
	fb, ipe_samps[order], kappas, prior=prior, smooth=f_smooth)

    # compute probability of falling
    newcond = "%s-%s-%s" % ("MO", fbtype, cond.split("-")[2])
    # newcond = "-".join(["MO"] + cond.split("-")[1:])
    p_outcomes = np.exp(mo.predict(
	model_theta[:-1], ipe_samps[order], kappas, f_smooth))

    # sample responses
    responses = np.random.rand(nfake, n_trial) < p_outcomes
    experiment[newcond] = pd.DataFrame(
	responses[:, undo_order], 
	columns=cols)

    # lat.plot_theta(
    # 	2, 3, cidx+1,
    # 	np.exp(model_theta),
    # 	cond,
    # 	exp=1.3,
    # 	cmap=cmap,
    # 	fontsize=14)
    # cidx += 1

conds = sorted(experiment.keys())
# lat.save("images/ideal_observer_static_beliefs.png", close=True)

# <codecell>

reload(lat)

nboot = 1000
nsamp_wc = 8
nsamp = 15
with_replacement = False

for cidx1, cond1 in enumerate(conds):
    arr1 = np.asarray(experiment[cond1])
    corrs = lat.bootcorr_wc(
	np.asarray(arr1), 
	nboot=nboot,
	nsamp=nsamp_wc,
	with_replacement=with_replacement)
    meancorr = np.mean(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond1, cond1, meancorr, semcorr)

    if cond1 == "old-all-10":
	continue

    for cidx2, cond2 in enumerate(conds[cidx1+1:]):
	if cond2 == "old-all-10":
	    continue
	if cond1.split("-")[-2:] != cond2.split("-")[-2:]:
	    continue
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

# <codecell>

rawipe0 = lat.load_model("stability")[1]
idx = np.nonzero(Stims[None, :] == rawoldhstim[:, None])[0]
nfell = (rawipe0[idx]['nfellA'] + rawipe0[idx]['nfellB']) / 10.

nanmean = scipy.stats.nanmean
nanstd = scipy.stats.nanstd

hdata = np.asarray(experiment['C-vfb-10'])
hmean = nanmean(hdata, axis=0)
hsem = nanstd(hdata, axis=0) / np.sqrt(hdata.shape[0])

sdata = np.asarray(experiment['MO-fb-10'])
smean = nanmean(sdata, axis=0)
ssem = nanstd(sdata, axis=0) / np.sqrt(sdata.shape[0])

print xcorr(hmean, smean)

plt.figure(3)
plt.clf()
plt.errorbar(hmean, smean, xerr=hsem, yerr=ssem, 
	     linestyle='', marker='o')
plt.xlabel("Human n=%d (visual feedback)" % hdata.shape[0])
plt.ylabel("Model n=%d (uniform prior, no learning)" % sdata.shape[0])
plt.title("Human vs. model judgments")

lat.save("images/human_v_model.png", close=False)

# <codecell>

fig = plt.figure(10)
plt.clf()
fig = plt.figure(11)
plt.clf()

allarrC = []
allarrE = []
allcondsC = []
allcondsE = []

for cond in conds:
    if cond.startswith("MO"):
	continue
    if cond.startswith("old"):
	continue
    if cond.endswith("nfb-10"):
	continue

    ratio = float(cond.split("-")[-1])
    arr = np.asarray(queries[cond]) == ratio
    index = np.array(queries[cond].columns, dtype='i8')
    idx = np.array(index)-6-np.arange(len(index))-1
    print idx
    binom = [scipy.stats.binom_test(x, arr.shape[0], 0.5) for x in np.sum(arr, axis=0)]
    print np.round(binom, decimals=8)
    mean = np.mean(arr, axis=0)
    sem = scipy.stats.sem(arr, axis=0, ddof=1)

    if cond.startswith("E-"):
	plt.figure(11)
	allarrE.append(arr)
	allcondsE.append(cond)
    else:
	plt.figure(10)
	allarrC.append(arr)
	allcondsC.append(cond)

    plt.errorbar(idx, mean, yerr=sem, 
		 label="%s r=%s (n=%d)"% (
		     cond.split("-")[1],
		     cond.split("-")[-1], 
		     arr.shape[0]))

for allarr, fignum, group in ((allarrC, 10, "C"), (allarrE, 11, "E")):
    arr = np.concatenate(allarr, axis=0)
    binom = [scipy.stats.binom_test(x, arr.shape[0], 0.5) for x in np.sum(arr, axis=0)]
    print np.round(binom, decimals=8)
    mean = np.mean(arr, axis=0)
    sem = scipy.stats.sem(arr, axis=0)

    plt.figure(fignum)
    plt.errorbar(idx, mean, yerr=sem, 
		 label="all (n=%d)"% (arr.shape[0]))

    plt.xlim(idx.min(), idx.max())
    plt.xticks(idx, idx)
    plt.xlabel("Trial")
    plt.ylabel("P(judge correct mass ratio)")
    plt.title("Explicit mass judgments (group %s)" % group)
    plt.legend(loc=0)
    
    lat.save("images/explicit_mass_judgments_%s.png" % group, close=False)
    

# <codecell>

for allarr, group in ((allarrC, "C"), (allarrE, "E")):
    arr = np.concatenate(allarr, axis=0)
    df = pd.DataFrame(
	np.array([np.sum(1-arr, axis=0), np.sum(arr, axis=0)]).T,
	index=idx,
	columns=["incorrect", "correct"])
    print group
    print df

    chi2, p, dof, ex = scipy.stats.chi2_contingency(df)
    print (chi2, p)
    print

# <codecell>

queries['E-fb-0.1']

# <codecell>


