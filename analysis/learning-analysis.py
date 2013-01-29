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
## Load data
######################################################################

reload(lat)

# human
training, posttest, experiment, queries = lat.load_turk(thresh=1)
hconds = sorted(experiment.keys())
for cond in hconds:
    print "%s: n=%d" % (cond, experiment[cond].shape[0])

# stims
Stims = np.array([
    x.split("~")[0] for x in zip(*experiment[experiment.keys()[0]].columns)[1]])

# model
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

n_trial      = Stims.size
n_fake_data  = 2000

f_smooth = True
p_ignore = 0.0

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
alpha = 0.2
#colors = ['r', '#AAAA00', 'g', 'c', 'b', 'm']
colors = ["r", "c", "k"]
#colors = cm.hsv(np.round(np.linspace(0, 220, n_cond)).astype('i8'))

# <codecell>

######################################################################
## Generate fake human data
######################################################################

reload(mo)
model_belief = {}
for cond in hconds:

    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    if obstype == "M" or fbtype == "vfb":
	continue

    cols = experiment[cond].columns
    order = np.argsort(zip(*cols)[0])
    undo_order = np.argsort(order)
    #nfake = experiment[cond].shape[0]
    nfake = n_fake_data

    # determine what feedback to give
    if fbtype == 'nfb':
	fb = np.empty((2, n_trial))*np.nan
	prior = np.zeros((2, n_kappas,))
	prior[0, :] = 1 # uniform
	prior[1, kappas.index(0.0)] = 1 # r=1.0
	prior = normalize(np.log(prior), axis=1)[1]
	# prior = None
    else:
	ridx = ratios.index(float(ratio))
	fb = feedback[:, order][ridx]
	prior = None

    responses, model_theta = mo.simulateResponses(
	nfake, fb, ipe_samps[order], kappas, 
	prior=prior, p_ignore=p_ignore, smooth=f_smooth)

    if fbtype == "nfb":
	newcond = "M-%s-%s-%s" % (group, fbtype, ratio)
	experiment[newcond] = pd.DataFrame(
	    responses[:, 0][:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta[0]
	
	newcond = "M-%s-%s-1" % (group, fbtype)
	experiment[newcond] = pd.DataFrame(
	    responses[:, 1][:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta[1]

    else:
	newcond = "M-%s-%s-%s" % (group, fbtype, ratio)
	experiment[newcond] = pd.DataFrame(
	    responses[:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta

    # explicit judgments
    if fbtype == "fb":
	cols = queries[cond].columns
	idx = np.array(cols)-6-np.arange(len(cols))-1
	theta = np.exp(model_theta[idx])
	if float(ratio) > 1:
	    pcorrect = np.sum(theta[:, ratios.index(1.0)+1:], axis=1)
	    other = 0.1
	else:
	    pcorrect = np.sum(theta[:, :ratios.index(1.0)], axis=1)
	    other = 10
	r = np.random.rand(nfake, 1) < pcorrect[None]
	responses = np.empty(r.shape)
	responses[r] = float(ratio)
	responses[~r] = other
	queries[newcond] = pd.DataFrame(
	    responses, columns=cols)
	# print newcond, pcorrect
	

# <codecell>

######################################################################
## Conditions
######################################################################

groups = ["C", "E", "all"]

cond_labels = {
    # order C
    'H-C-nfb-10': 'No feedback C',
    'H-C-vfb-0.1': 'Visual feedback C (r=0.1)',
    'H-C-vfb-10': 'Visual feedback C (r=10)',
    'H-C-fb-0.1': 'Text feedback C (r=0.1)',
    'H-C-fb-10': 'Text feedback C (r=10)',

    'M-C-nfb-10': 'Fixed observer C (uniform)',
    'M-C-nfb-1': 'Fixed observer C (r=1.0)',
    'M-C-fb-0.1': 'Learning observer C (r=0.1)',
    'M-C-fb-10': 'Learning observer C (r=10)',

    # order E
    'H-E-nfb-10': 'No feedback E',
    'H-E-vfb-0.1': 'Visual feedback E (r=0.1)',
    'H-E-vfb-10': 'Visual feedback E (r=10)',
    'H-E-fb-0.1': 'Text feedback E (r=0.1)',
    'H-E-fb-10': 'Text feedback E (r=10)',

    'M-E-nfb-10': 'Fixed observer E (uniform)',
    'M-E-nfb-1': 'Fixed observer E (r=1.0)',
    'M-E-fb-0.1': 'Learning observer E (r=0.1)',
    'M-E-fb-10': 'Learning observer E (r=10)',

    # all orders
    'H-all-nfb-10': 'No feedback all',
    'H-all-vfb-0.1': 'Visual feedback all (r=0.1)',
    'H-all-vfb-10': 'Visual feedback all (r=10)',
    'H-all-fb-0.1': 'Text feedback all (r=0.1)',
    'H-all-fb-10': 'Text feedback all (r=10)',

    'M-all-nfb-10': 'Fixed observer E (uniform)',
    'M-all-nfb-1': 'Fixed observer E (r=1.0)',
    'M-all-fb-0.1': 'Learning observer all (r=0.1)',
    'M-all-fb-10': 'Learning observer all (r=10)',
    }

conds = [
    # order C
    'H-C-fb-0.1',
    'H-C-vfb-0.1',
    'M-C-fb-0.1',

    'H-C-fb-10',
    'H-C-vfb-10',
    'M-C-fb-10',

    'H-C-nfb-10',
    'M-C-nfb-10',
    'M-C-nfb-1',

    # order E
    'H-E-fb-0.1',
    'H-E-vfb-0.1',
    'M-E-fb-0.1',

    'H-E-fb-10',
    'H-E-vfb-10',
    'M-E-fb-10',

    'H-E-nfb-10',
    'M-E-nfb-10',
    'M-E-nfb-1',

    # all orders
    'H-all-fb-0.1',
    'H-all-vfb-0.1',
    'M-all-fb-0.1',

    'H-all-fb-10',
    'H-all-vfb-10',
    'M-all-fb-10',

    'H-all-nfb-10',
    'M-all-nfb-10',
    'M-all-nfb-1',
    ]
n_cond = len(conds)

# cond_labels = dict([(c, c) for c in conds])
# conds = sorted(np.unique(["-".join(c.split("-")[:-1]) for c in conds]))
    

# <codecell>

######################################################################
## Compute likelihoods under various models
######################################################################

reload(lat)

ir1 = list(kappas).index(0.0)
ir10 = list(kappas).index(1.0)
ir01 = list(kappas).index(-1.0)

# random model
model_random = lat.CI(lat.random_model_lh(conds, n_trial), conds)

# fixed models
theta = np.log(np.eye(n_kappas))
fb = np.empty((n_kappas, n_trial))*np.nan
model_fixed = lat.CI(lat.block_lh(
    experiment, fb, ipe_samps, theta, kappas, 
    f_smooth=f_smooth, p_ignore=p_ignore), conds)
model_true01 = dict([(cond, model_fixed[cond][ir01]) for cond in model_fixed])
model_true1 = dict([(cond, model_fixed[cond][ir1]) for cond in model_fixed])
model_true10 = dict([(cond, model_fixed[cond][ir10]) for cond in model_fixed])
model_uniform = lat.CI(lat.block_lh(
    experiment, nofeedback, ipe_samps, None, kappas,
    f_smooth=f_smooth, p_ignore=p_ignore), conds)
	
# learning models
model_learn = lat.CI(lat.block_lh(
    experiment, feedback[[ir01, ir10]], ipe_samps, None, kappas, 
    f_smooth=f_smooth, p_ignore=p_ignore), conds)
model_learn01 = dict([(cond, model_learn[cond][0]) for cond in model_fixed])
model_learn10 = dict([(cond, model_learn[cond][1]) for cond in model_fixed])

# all the models
mnames = np.array([
	"random",
	"fixed 0.1",
	"learning 0.1",
	"fixed 10",
	"learning 10"
	"fixed uniform",
	"fixed r=1",
	])
mparams = np.array([0, 1, 2, 1, 2, 1, 1])
models = [
    model_random,
    model_true01,
    model_learn01,
    model_true10,
    model_learn10,
    model_uniform,
    model_true1,
    ]

# <codecell>

######################################################################
## Plot explicit judgments
######################################################################

reload(lat)

fig = plt.figure(10)
plt.clf()

emjs = {}
for cond in conds:
    if cond not in queries:
	continue
    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    emjs[cond] = np.asarray(queries[cond]) == float(ratio)
    index = np.array(queries[cond].columns, dtype='i8')
    X = np.array(index)-6-np.arange(len(index))-1
emj_stats = lat.CI(emjs, conds)

for i, group in enumerate(groups):

    plt.subplot(1, 3, i+1)
    idx = 0

    for cidx, cond in enumerate(conds):
	if cond not in emj_stats:
	    continue
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if (grp != group):
	    continue
	mean, lower, upper, logsums, sums, n = emj_stats[cond].T
	mean = np.exp(mean)
	sem = mean - np.exp(lower)

	color = colors[(idx/3) % len(colors)]
	if obstype == "M":
	    linestyle = '-'
	elif fbtype == "fb":
	    linestyle = '-.'
	elif fbtype in ("vfb", "nfb"):
	    linestyle = '--'

	label = "%s %s r=%s (n=%d)" % (
	    obstype, fbtype, ratio, n[0])
	plt.errorbar(X, mean, yerr=sem, label=label, 
		     linewidth=2, color=color, linestyle=linestyle)

	idx += 1

    # arr = np.concatenate(allarr, axis=0)
    # lat.plot_explicit_judgments(idx, arr)
	
    plt.xlim(X.min(), X.max())
    plt.xticks(X, X)
    plt.xlabel("Trial")
    plt.ylabel("P(judge correct mass ratio)")
    plt.title("Explicit mass judgments (group %s)" % group)
    plt.legend(loc=0, fontsize=10)

# allarr = []
# plt.subplot(1, 3, 3)

# for group, fbtype, ratio in sorted(allgroups.keys()):
#     arr = np.concatenate(allgroups[group, fbtype, ratio], axis=0)
#     if group == "Human":
# 	allarr.append(arr)
#     lat.plot_explicit_judgments(idx, arr, "%s %s" % (group, fbtype), ratio)

# arr = np.concatenate(allarr, axis=0)
# lat.plot_explicit_judgments(idx, arr)
	
# plt.xlim(idx.min(), idx.max())
# plt.xticks(idx, idx)
# plt.xlabel("Trial")
# plt.ylabel("P(judge correct mass ratio)")
# plt.title("Explicit mass judgments (all)")
# plt.legend(loc=0, fontsize=10)

fig = plt.gcf()
fig.set_figwidth(12)
fig.set_figheight(4)

lat.save("images/explicit_mass_judgments.png", close=False)
    

# <codecell>

######################################################################
## Binomial analysis of explicit judgments
######################################################################

reload(lat)
for i, group in enumerate(groups):
    allarr = []

    for cidx, cond in enumerate(conds):
	if cond not in emj_stats:
	    continue
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if (grp != group):
	    continue
	mean, lower, upper, logsums, sums, n = emj_stats[cond].T

	# arr = np.asarray(queries[cond]) == float(ratio)
	binom = [scipy.stats.binom_test(x, n[0], 0.5) for x in sums]

        print cond
        print "  ", np.round(binom, decimals=3)

    # arr = np.concatenate(allarr, axis=0)
    # binom = [scipy.stats.binom_test(x, arr.shape[0], 0.5) 
    # 	     for x in np.sum(arr, axis=0)]
    # print "All %s" % group
    # print "  ", np.round(binom, decimals=3)
    

# <codecell>

######################################################################
## Chi-square analysis of explicit judgments
######################################################################

for i, group in enumerate(groups):
    allarr = []

    for cond in conds:
	if cond not in queries:
	    continue
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if grp != group or obstype == "M":
	    continue
	print cond

	arr = np.asarray(queries[cond]) == float(ratio)
	allarr.append(arr)

    if allarr == []:
	continue

    arr = np.concatenate(allarr, axis=0)
    df = pd.DataFrame(
	np.array([np.sum(1-arr, axis=0), np.sum(arr, axis=0)]).T,
	index=X,
	columns=["incorrect", "correct"])
    print group
    print df

    chi2, p, dof, ex = scipy.stats.chi2_contingency(df)
    print (chi2, p)
    print

# <codecell>

######################################################################
## Plot smoothed likelihoods
######################################################################

reload(lat)
fig = plt.figure(1)
plt.clf()
istim = [0, 1]
print Stims[istim]
lat.plot_smoothing(ipe_samps, Stims, istim, kappas)

lat.save("images/likelihood_smoothing.png", close=True)

# <codecell>

######################################################################
## Plot ideal learning observer beliefs
######################################################################

reload(lat)
beliefs = {}
for cond in conds:
    if cond not in model_belief:
	continue
    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    if obstype == "M" and fbtype not in ("vfb", "nfb"):
	beliefs[cond] = model_belief[cond]

lat.plot_belief(2, 2, 2, beliefs, kappas, cmap)
lat.save("images/ideal_observer_beliefs.png", close=True)

# <codecell>

######################################################################
## Plot likelihoods under fixed models
######################################################################

reload(lat)
x = np.arange(n_kappas)

for i, group in enumerate(groups):
    idx = 0
    plt.figure(30+i)
    plt.clf()

    for cidx, cond in enumerate(conds):
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	
	if grp != group:
	    continue
		
	color = colors[(idx/3) % len(colors)]
	if obstype == "M":
	    linestyle = '-'
	elif fbtype == "fb":
	    linestyle = '-.'
	else:
	    linestyle = '--'
	if obstype == "M" and fbtype == "nfb" and ratio == "1":
	    color = "#996600"
	    
	mean, lower, upper, sumlogs, sums, n = model_fixed[cond].T
	plt.fill_between(x, lower, upper, color=color, alpha=0.2)
	plt.plot(x, mean, label=cond_labels[cond], color=color, linewidth=2,
		 linestyle=linestyle)
	# plt.plot(x, sums[cidx], label=cond_labels[cond], color=color, linewidth=2,
	# 	     linestyle=linestyle)

	idx += 1

    plt.xticks(x, ratios, rotation=90)
    plt.xlabel("Fixed model mass ratio")
    plt.ylabel("Log likelihood of responses")
    plt.legend(loc=4, ncol=2, fontsize=9)
    plt.xlim(x[0], x[-1])
    plt.ylim(-35, -20)

    plt.suptitle("Likelihood of responses under fixed models (%s)" % group)
    fig.set_figwidth(8)
    fig.set_figheight(6)

    lat.save("images/fixed_model_performance_%s.png" % group, close=False)

# <codecell>

######################################################################
## Plot likelihoods under other models
######################################################################

n = n_cond / len(groups)
width = 0.7 / n
x0 = np.arange(len(models))

for i, group in enumerate(groups):
    idx = 0
    plt.figure(40+i)
    plt.clf()
    
    for cidx, cond in enumerate(conds):
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if grp != group:
	    continue

	height = np.array([models[x][cond][0] for x in xrange(len(models))])
	err = np.array([np.abs(models[x][cond][0] - models[x][cond][[1,2]])
			for x in xrange(len(models))])

	color = colors[(idx/3) % len(colors)]
	if obstype == "M":
	    alpha = 1.0
	elif fbtype == "vfb":
	    alpha = 0.4
	elif fbtype in ("fb", "nfb"):
	    alpha = 0.2
	if obstype == "M" and fbtype == "nfb" and ratio == "1":
	    color = "#996600"

	x = x0 + width*(idx-(n/2.)) + (width/2.)
	plt.bar(x, height, yerr=err.T, 
		color=color,
		ecolor='k', align='center', width=width, 
		label=cond_labels[cond], alpha=alpha)

	idx += 1

    plt.xticks(x0, mnames)
    plt.ylim(-35, -20)
    plt.xlim(x0.min()-0.5, x0.max()+0.5)
    plt.legend(loc=0, ncol=2, fontsize=9)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Log likelihood", fontsize=14)

    plt.suptitle(
	    "Likelihood of human and ideal observer judgments (%s)" % group, 
	    fontsize=16)
    fig.set_figwidth(8)
    fig.set_figheight(6)

    lat.save("images/model_performance_%s.png" % group, close=False)

# <codecell>


reload(lat)
window = 10

lh = np.empty((n_trial-window, n_kappas, len(conds), 4))
for t in xrange(0, n_trial-window):
    t0 = t
    t1 = t+window
    thetas = np.eye(n_kappas)
    thetas = normalize(np.log(thetas), axis=1)[1]
    fb = np.empty((n_kappas, n_trial)) * np.nan
    lh[t] = lat.CI(lat.block_lh(
	experiment, fb, ipe_samps, thetas, 
	kappas, t0, t1, 
	f_smooth=f_smooth, p_ignore=p_ignore), conds)

# <codecell>

# best = np.argmax(lh[..., -1], axis=1).T
x = np.arange(0, n_trial, 1)

plt.figure(100)
plt.clf()

for cidx, cond in enumerate(conds):
    color = colors[(cidx/3) % len(colors)]
    if cond.startswith("MO"):
	linestyle = '-'
    elif cond.split("-")[1] == "fb":
	linestyle = '-.'
    else:
	linestyle = "--"
    lat.plot_theta(
	6, 3, cidx+1, 
	np.exp(normalize(lh[:, :, cidx, -1], axis=1)[1]), 
	cond_labels[conds[cidx]], exp=np.e)
    x = np.round(np.arange(0, n_trial-window, 5)).astype('i8')
    plt.xticks(x, x+window)

fig = plt.gcf()
fig.set_figwidth(12)
fig.set_figheight(10)

plt.subplots_adjust(hspace=0.5)
plt.suptitle("Fixed model log likelihoods over time (window of %d)" % window)

lat.save("images/sliding-windows.png", close=False)

# <codecell>

# BIC: -2*ln(L) + k*ln(n)
# L : maximized likelihood function
# k : number of parameters
# n : sample size

def samplesize(data):
    sizes = []
    for cond in conds:
	d = data[cond]
	sizes.append(d.shape[0])
    return np.array(sizes)

L = models[0]
k = np.array(mparams)[None]
#k = np.zeros(L.shape)
n = samplesize(experiment)[:, None]

BIC = -2*L + k*np.log(n)
best = np.argmin(BIC, axis=1)
zip(conds, mnames[best])


# <codecell>


# <codecell>


