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

reload(lat)
fig = plt.figure(1)
plt.clf()
lat.plot_smoothing(ipe_samps, Stims, 6, kappas)
fig.set_figheight(6), 
fig.set_figwidth(8)

lat.save("images/likelihood_smoothing.png", close=True)

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

######################################################################
## Generate fake human data
######################################################################

fig = plt.figure(2)
plt.clf()
plt.suptitle("Ideal Observer Beliefs")
cidx = 0

reload(mo)
nfake = 2000
for cond in sorted(experiment.keys()):

    group, fbtype, ratio, cb = lat.parse_condition(cond)
    if group == "MO":
	continue
    if fbtype == "vfb":
	fbtype = "fb"

    cols = experiment[cond].columns
    order = np.argsort(zip(*cols)[0])
    undo_order = np.argsort(order)
    #nfake = experiment[cond].shape[0]

    # determine what feedback to give
    if fbtype == 'nfb':
	fb = nofeedback[..., order]
	# prior = np.zeros((n_kappas,))
	# prior[kappas.index(0.0)] = 1
	# prior = normalize(np.log(prior))[1]
	prior = None
    else:
	ridx = ratios.index(ratio)
	fb = feedback[:, order][ridx]
	prior = None

    #newcond = "-".join(["MO"] + cond.split("-")[1:])
    newcond = "%s-%s-%s" % ("MO", fbtype, cond.split("-")[2])
    responses, model_theta = mo.simulateResponses(
	nfake, fb, ipe_samps[order], kappas, 
	prior=prior, p_ignore=p_ignore, smooth=f_smooth)
    experiment[newcond] = pd.DataFrame(
	responses[:, undo_order], 
	columns=cols)

    lat.plot_theta(
	3, 2, cidx+1,
	np.exp(model_theta),
	cond,
	exp=1.3,
	cmap=cmap,
	fontsize=14)
    cidx += 1
    
lat.save("images/ideal_observer_beliefs.png", close=True)

# <codecell>

cond_labels = {
    'C-nfb-10': 'No-feedback',
    'C-vfb-0.1': 'Visual feedback (r=0.1)',
    'C-vfb-10': 'Visual feedback (r=10)',
    'C-fb-0.1': 'Text feedback (r=0.1)',
    'C-fb-10': 'Text feedback (r=10)',
    'MO-nfb-10': 'Uniform fixed observer',
    'MO-fb-0.1': 'Learning observer (r=0.1)',
    'MO-fb-10': 'Learning observer (r=10)',
    }

conds = [
    'C-fb-0.1',
    'C-vfb-0.1',
    'MO-fb-0.1',
    'C-fb-10',
    'C-vfb-10',
    'MO-fb-10',
    'C-nfb-10',
    'MO-nfb-10',
    ]
n_cond = len(conds)

# cond_labels = dict([(c, c) for c in conds])
# conds = sorted(np.unique(["-".join(c.split("-")[:-1]) for c in conds]))
    

# <codecell>

# bootstrapped correlations
reload(lat)

nboot = 1000
nsamp = 5
with_replacement = False

for cond in conds:

    arr = np.asarray(experiment[cond])

    corrs = lat.bootcorr_wc(
	arr,
	nboot=nboot,
	nsamp=nsamp,
	with_replacement=with_replacement)
    meancorr = np.mean(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond, cond, meancorr, semcorr)

# <codecell>

def CI(data):
    bmvs = scipy.stats.bayes_mvs
    stats = []
    for cidx, cond in enumerate(conds):
	shape = data[cond].shape
	assert len(shape) == 2
	info = []
	for i in xrange(shape[1]):
	    shape = data[cond][:, i].shape
	    if shape == (1,):
		mean = lower = upper = sum = np.log(data[cond][:, i][0])
	    else:
		#mean, (lower, upper) = bmvs(data[cond][:, i])[0]
		mean = np.mean(data[cond][:, i])
		sem = scipy.stats.sem(data[cond][:, i])
		lower = np.log(mean - sem)
		upper = np.log(mean + sem)
		mean = np.log(mean)
		sum = np.sum(np.log(data[cond][:, i]))
	    info.append([mean, lower, upper, sum])
	stats.append(info)
    stats = np.swapaxes(np.array(stats), 0, 1)
    return stats
	
	

# <codecell>

ir1 = list(kappas).index(0.0)
ir10 = list(kappas).index(1.0)
ir01 = list(kappas).index(-1.0)
reload(lat)

# random model
model_random, = CI(lat.random_model_lh(conds, n_trial))

# <codecell>

reload(lat)

# fixed models
thetas = np.zeros((3, n_kappas))
thetas[0, :] = 1
thetas[1, ir10] = 1
thetas[2, ir01] = 1
thetas = normalize(np.log(thetas), axis=1)[1]
fb = np.empty((3, n_trial))*np.nan

model_uniform, model_true10, model_true01 = CI(lat.block_lh(
    experiment, fb, ipe_samps, thetas, kappas, 
    f_smooth=f_smooth, p_ignore=p_ignore))
	

# <codecell>

reload(lat)

# learning models
model_learn01, model_learn10 = CI(lat.block_lh(
    experiment, feedback[[ir01, ir10]], ipe_samps, None, kappas, 
    f_smooth=f_smooth, p_ignore=p_ignore))

# <codecell>

# all the models
mnames = np.array([
	"random",
	"fixed 0.1",
	"learning 0.1",
	"fixed uniform",
	"fixed 10",
	"learning 10"
	])
mparams = np.array([0, 1, 2, 1, 1, 2])

models = np.concatenate([
    model_random[None],
    model_true01[None],
    model_learn01[None],
    model_uniform[None],
    model_true10[None],
    model_learn10[None]
    ], axis=0).T

# <codecell>

reload(lat)
theta = np.log(np.eye(n_kappas))
fb = np.empty((n_kappas, n_trial))*np.nan

mean, lower, upper, sums = CI(lat.block_lh(
    experiment, fb, ipe_samps, theta, kappas, f_smooth=f_smooth,
    p_ignore=p_ignore)).T
# sums, = CI(lat.block_lh(
#     experiment, fb, ipe_samps, theta, kappas, f_smooth=f_smooth,
#     f_average=False, f_round=False))

x = np.arange(n_kappas)
fig = plt.figure(3)
plt.clf()

for cidx, cond in enumerate(conds):
    color = colors[(cidx/3) % len(colors)]
    if cond.startswith("MO"):
	linestyle = '-'
    elif cond.split("-")[1] == "fb":
	linestyle = '-.'
    else:
	linestyle = '--'
    plt.fill_between(x, lower[cidx], upper[cidx], color=color, alpha=alpha)
    plt.plot(x, mean[cidx], label=cond_labels[cond], color=color, linewidth=2,
    	     linestyle=linestyle)
    # plt.plot(x, sums[cidx], label=cond_labels[cond], color=color, linewidth=2,
    # 	     linestyle=linestyle)

plt.xticks(x, ratios, rotation=90)
plt.xlabel("Fixed model mass ratio")
plt.ylabel("Log likelihood of responses")
plt.legend(loc=4, ncol=2, fontsize=12)
plt.xlim(x[0], x[-1])
plt.ylim(-34, -20)
plt.title("Likelihood of responses under fixed models")
fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/fixed_model_performance.png", close=False)

# <codecell>

# plot model performance
x0 = np.arange(models.shape[2])
height = models[0]
err = np.abs(models[[0]] - models[[1,2]])
width = 0.7 / n_cond
fig = plt.figure(4)
plt.clf()

for cidx, cond in enumerate(conds):
    color = colors[(cidx/3) % len(colors)]
    if cond.startswith("MO"):
	alpha = 1.0
    elif cond.split("-")[1] == "vfb":
	alpha = 0.4
    else:
	alpha = 0.2
    x = x0 + width*(cidx-(n_cond/2.)) + (width/2.)
    plt.bar(x, height[cidx], yerr=err[:, cidx], 
	    color=color,
	    ecolor='k', align='center', width=width, 
	    label=cond_labels[cond], alpha=alpha)

plt.xticks(x0, [
    "Random", 
    "Fixed\nr=0.1",
    "Learning\nr=0.1",
    "Fixed\nr=uniform", 
    "Fixed\nr=10.0", 
    "Learning\nr=10.0"
    ])
#plt.ylim(int(np.min(height-err))-1, int(np.max(height))+1)
plt.ylim(-34, -20)
plt.xlim(x0.min()-0.5, x0.max()+0.5)
plt.legend(loc=0, ncol=2, fontsize=12)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Log likelihood of responses, $\Pr(J|S,B)$", fontsize=14)
plt.title("Likelihood of human and ideal observer judgments", fontsize=16)

fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/model_performance.png", close=False)

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


reload(lat)
window = 10
nsteps = n_trial / window

lh = np.empty((n_trial, n_kappas, len(conds), 4))
for t in xrange(1, n_trial):
    t0 = max(0, t-window)
    t1 = t
    thetas = np.eye(n_kappas)
    thetas = normalize(np.log(thetas), axis=1)[1]
    fb = np.empty((n_kappas, n_trial)) * np.nan
    lh[t] = CI(lat.block_lh(
	experiment, fb, ipe_samps, thetas, 
	kappas, t0, t1, 
	f_smooth=f_smooth, p_ignore=p_ignore))

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
	3, 3, cidx+1, 
	np.exp(normalize(lh[:, :, cidx, -1], axis=1)[1]), 
	cond_labels[conds[cidx]])
    # plt.plot(x, np.array(kappas)[best[cidx]], label=cond_labels[cond], 
    # 	     color=color, linestyle=linestyle, linewidth=2)

#plt.yticks(kappas, ratios)
#plt.grid(True)
#plt.legend()

plt.suptitle("Fixed model log likelihoods over time (window of %d)" % window)

lat.save("images/sliding-windows.png", close=False)

# <codecell>


