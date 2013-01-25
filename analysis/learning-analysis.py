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
    Stims = np.array(sorted([
	x.split("~")[0] for x in fh.read().strip().split("\n") if x != ""]))

# <codecell>

######################################################################
## Load human data
######################################################################

reload(lat)
training, posttest, experiment, queries = lat.load_turk_learning()

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
    if fbtype == 'nfb':
	fb = nofeedback[order][:, None]
    else:
	ridx = ratios.index(ratio)
	fb = feedback[order][:, [ridx]][:, None]
    
    # learning model beliefs
    model_lh, model_joint, model_theta = mo.ModelObserver(
	ipe_samps[order], fb,
	outcomes=None,
	initial=None,
	respond=False,
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
    responses = np.random.rand(nfake, n_trial) < p_outcomes
    experiment[newcond] = pd.DataFrame(
	responses[:, undo_order], 
	columns=cols)

    lat.plot_theta(
	1, 6, cidx+1,
	np.exp(model_theta[0]),
	cond,
	exp=1.3,
	cmap=cmap,
	fontsize=14)
    cidx += 1
    
lat.save("images/ideal_observer_beliefs.png", close=False)

# <codecell>

cond_labels = {
    # 'A-nfb': 'No feedback',
    # 'A-fb': 'Feedback',
    # 'A-fb-ideal0.1-mo': 'Learning observer, r=0.1',
    # 'A-fb-ideal10-mo': 'Learning observer, r=10',
    # 'A-fb-fixed0.1-mo': 'Fixed Observer, r=0.1',
    # 'A-fb-fixed10-mo': 'Fixed Observer, r=10'
    'B-nfb-10': 'Human no-feedback',
    'B-fb-0.1': '(r=0.1) Human feedback',
    'B-fb-10': '(r=10) Human feedback',
    'MO-nfb-10': 'Uniform fixed observer',
    # 'MO-nfb-1.0': '(r=1) Fixed observer',
    'MO-fb-0.1': '(r=0.1) Learning observer',
    'MO-fb-10': '(r=10) Learning observer',
    # 'MO-nfb-0.1': '(r=0.1) Fixed observer',
    # 'MO-nfb-10': '(r=10) Fixed observer',
    }

conds = experiment.keys()
newconds = [
    'B-fb-0.1',
    'MO-fb-0.1',
    # 'MO-nfb-0.1',
    'B-fb-10',
    'MO-fb-10',
    'MO-nfb-10',
    'B-nfb-10',
    # 'MO-nfb',
    # 'MO-nfb-1.0',
    ]
n_cond = len(newconds)

# cond_labels = dict([(c, c) for c in conds])
# newconds = sorted(np.unique(["-".join(c.split("-")[:-1]) for c in conds]))
    

# <codecell>

# bootstrapped correlations
reload(lat)

nboot = 1000
nsamp = 5
with_replacement = False

for cond in newconds:

    arr1 = np.asarray(experiment[cond + '-cb0'])
    arr2 = np.asarray(experiment[cond + '-cb1'])
    arr3 = np.vstack([arr1, arr2])

    corrs = lat.bootcorr(
	np.asarray(arr1), np.asarray(arr2), 
	nboot=nboot, 
	nsamp=nsamp,
	with_replacement=with_replacement)
    meancorr = np.mean(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond+"-cb0", cond+"-cb1", meancorr, semcorr)

    corrs = lat.bootcorr_wc(
	np.asarray(arr3), 
	nboot=nboot,
	nsamp=nsamp,
	with_replacement=with_replacement)
    meancorr = np.mean(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond, cond, meancorr, semcorr)

# <codecell>

def collapse(data):
    stacked = [np.vstack([data[c] for c in conds if c.startswith(nc+"-cb")]) for nc in newconds]
    means = np.array([np.mean(x, axis=0) for x in stacked])
    sems = np.array([scipy.stats.sem(x, axis=0) for x in stacked])
    sems[np.isnan(sems)] = 0
    mean = np.log(means)
    lower = np.log(means - sems)
    upper = np.log(means + sems)
    out = np.array([mean, lower, upper]).T
    return out
	

# <codecell>

ir1 = list(kappas).index(0.0)
ir10 = list(kappas).index(1.0)
ir01 = list(kappas).index(-1.0)
reload(lat)

# random model
model_random = collapse(lat.random_model_lh(conds, n_trial))[0]

# <codecell>

reload(lat)

# fixed models
thetas = np.zeros((3, n_trial+1, n_kappas))
thetas[0, :, ir1] = 1
thetas[1, :, ir10] = 1
thetas[2, :, ir01] = 1
model_same, model_true10, model_true01 = collapse(lat.fixed_model_lh(
    conds, experiment, ipe_samps, np.log(thetas), f_smooth=f_smooth))
	

# <codecell>

reload(lat)

# learning models
model_learn01, model_learn10 = collapse(lat.learning_model_lh(
    conds, experiment, ipe_samps, feedback, [ir01, ir10], f_smooth=f_smooth))

# <codecell>

# all the models
models = np.array([
    model_random,
    model_true01,
    model_learn01,
    # model_same,
    model_true10,
    model_learn10
    ]).T

# <codecell>

theta = np.log(np.eye(n_kappas)[:, None] * np.ones((n_kappas, n_trial+1, n_kappas)))
mean, upper, lower = collapse(lat.fixed_model_lh(
    conds, experiment, ipe_samps, theta, f_smooth=f_smooth)).T

x = np.arange(n_kappas)
fig = plt.figure(3)
plt.clf()

colors = ['r', '#FF9966', '#AAAA00', 'g', 'c', 'b', 'm']
for cidx, cond in enumerate(newconds):
    color = colors[cidx % len(colors)]
    if cidx >= len(colors):
	linestyle = '--'
    else:
	linestyle = '-'
    plt.fill_between(x, lower[cidx], upper[cidx], color=color, alpha=alpha)
    plt.plot(x, mean[cidx], label=cond_labels[cond], color=color, linewidth=2,
	     linestyle=linestyle)

plt.xticks(x, ratios, rotation=90)
plt.xlabel("Fixed model mass ratio")
plt.ylabel("Log likelihood of responses")
plt.legend(loc=4, ncol=2)
plt.xlim(x[0], x[-1])
plt.ylim(-24, -8)
plt.title("Likelihood of responses under fixed models")
fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/fixed_model_performance.png", close=False)

# <codecell>

# plot model performance
x0 = np.arange(models.shape[2])
height = models[0]
err = np.abs(models[[0]] - models[1:])
width = 0.7 / n_cond
fig = plt.figure(4)
plt.clf()

for cidx, cond in enumerate(newconds):
    color = colors[cidx % len(colors)]
    x = x0 + width*(cidx-(n_cond/2.)) + (width/2.)
    plt.bar(x, height[cidx], yerr=err[:, cidx], color=color,
	    ecolor='k', align='center', width=width, label=cond_labels[cond])

plt.xticks(x0, [
    "Random", 
    "Fixed\nr=0.1",
    "Learning\nr=0.1",
    # "Fixed\nr=1.0", 
    "Fixed\nr=10.0", 
    "Learning\nr=10.0"
    ])
#plt.ylim(int(np.min(height-err))-1, int(np.max(height))+1)
plt.ylim(-25, -5)
plt.xlim(x0.min()-0.5, x0.max()+0.5)
plt.legend(loc=0, ncol=2)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Log likelihood of responses, $\Pr(J|S,B)$", fontsize=14)
plt.title("Likelihood of human and ideal observer judgments", fontsize=16)

fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/model_performance.png", close=False)

# <codecell>

thetas = normalize(np.log(np.ones((1, n_trial+1, n_kappas))), axis=2)[1]
window = 5
nsteps = n_trial / window
lh = np.empty((nsteps, len(newconds), 3))
x = np.arange(nsteps)
for t in xrange(nsteps):
    lh[t] = collapse(lat.fixed_model_lh(
	conds, experiment, ipe_samps, thetas, t*window, (t+1)*window, f_smooth))[0]

# <codecell>

fig = plt.figure(5)
plt.clf()

for cidx, cond in enumerate(newconds):
    color = colors[cidx % len(colors)]
    if cidx >= len(colors):
	linestyle = '--'
    else:
	linestyle = '-'
    plt.fill_between(x, lh[:, cidx, 1], lh[:, cidx, 2],
		     color=color, alpha=alpha)
    plt.plot(lh[:, cidx, 0], color=color, label=cond_labels[cond],
	     linestyle=linestyle)

	
plt.legend(loc=0, ncol=2)
fig.set_figwidth(8)
fig.set_figheight(6)
plt.xlim(0, nsteps-1)
plt.xticks(np.arange(0, nsteps), np.linspace(window, n_trial, nsteps))
plt.xlabel("Trial")
plt.ylabel("Likelihood")
plt.title("Likelihood of observer responses, averaged over trial blocks")
plt.ylim(-5.5, -1.5)

# <codecell>

def calc_correct(kidx, window):

    correct = {}
    for cidx, cond in enumerate(conds):
	order = np.argsort(zip(*experiment[cond].columns)[0])

	true = feedback[:, kidx, 0][order]
	resp = np.asarray(experiment[cond])

	correct[cond] = np.mean(np.array(np.split((resp==true).T, n_trial/window)), axis=1).T

    CC = collapse(correct)
    return CC

# <codecell>



window = 5

for i, r in enumerate([['0.1', '1.0', 'nfb'], ['10', '1.0', 'nfb']]):
    plt.figure(6+i)
    plt.clf()

    i = 0
    for cidx, cond in enumerate(newconds):
	if cond.split("-")[-1] not in r:
	    continue
	x = np.arange(window, n_trial+1, window)
	CC = calc_correct(list(ratios).index(float(r[0])), window)
	color = colors[i % len(colors)]
	if i >= len(colors):
	    linestyle = '--'
	else:
	    linestyle = '-'
	plt.fill_between(x, CC[:, cidx, 1], CC[:, cidx, 2],
			 color=color, alpha=alpha)
	plt.plot(x, CC[:, cidx, 0], color=color, label=cond_labels[cond],
		 linestyle=linestyle)
	i += 1

	
    plt.legend(loc=0, ncol=2)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    plt.xlim(window, n_trial)
    plt.xticks(x, x)
    plt.xlabel("Trial")
    plt.ylabel(r"$e^{L/n}$")
    plt.title("Likelihood of observer responses, averaged over trial blocks")
    #plt.ylim(0.5, 1)

# <codecell>

p_response_mean = np.mean(p_responses, axis=2)
p_response_sem = scipy.stats.sem(p_responses, axis=2)

x = np.arange(n_trial)
upper = p_response_mean + p_response_sem
lower = p_response_mean - p_response_sem
mean = p_response_mean

#clr = ['r', 'b']
k = 0 
def plot(i, j, label):
    global k
    plt.fill_between(x, lower[i, j], upper[i, j], color=colors[k], alpha=alpha)
    plt.plot(mean[i, j], color=colors[k], label=label)
    k += 1

plot(0, list(kappas).index(-1.0), "learning, r=0.1")
plot(1, list(kappas).index(-1.0), "fixed, r=0.1")
plot(2, 0, "fixed, uniform")
plot(0, list(kappas).index(1.0), "learning, r=10")
plot(1, list(kappas).index(1.0), "fixed, r=10")
	
plt.legend(loc=0)
fig = plt.gcf()
fig.set_figwidth(8)
fig.set_figheight(6)
plt.xlim(0, n_trial-1)
plt.xlabel("Trial")
plt.ylabel("Likelihood")
plt.title("Likelihood of observer responses, averaged over trial orderings")
plt.ylim(0.45, 0.6)

lat.save("images/likelihoods_over_time.png", close=False)

# <codecell>

# model_thetas = np.empty((nsamp, n_kappas, n_trial+1, n_kappas))
# for i in xrange(model_thetas.shape[0]):
#     if i%10 == 0:
# 	print i
#     model_lh, model_joint, model_theta = mo.ModelObserver(
# 	ipe_samps[orders[i]],
# 	feedback[orders[i]][:, None],
# 	outcomes=None,
# 	respond=False,
# 	p_ignore_stimulus=p_ignore_stimulus,
# 	smooth=f_smooth)
#     model_thetas[i] = model_theta

# <codecell>


# p_responses = np.empty((3, n_kappas, nsamp, n_trial))
# theta_fixed = np.log(np.eye(n_kappas))

# theta_uniform = normalize(np.log(np.ones(n_kappas)))[1]
# sdata = np.asarray(ipe)

# for i in xrange(nsamp):
#     if i%10 == 0:
# 	print i
#     sd = sdata[ikappa, orders[i]]
#     for t in xrange(n_trial):
# 	# learning
# 	p_outcomes = np.exp(mo.predict(
# 	    model_thetas[i, :, t],
# 	    outcomes[:, None], 
# 	    ipe_samps[orders[i]][t],
# 	    f_smooth))[:, 1]
# 	resp = np.random.rand() < p_outcomes
# 	p_responses[0, :, i, t] = (resp * sd[t]) + ((1-resp) * (1-sd[t]))

# 	# fixed at true ratio
# 	p_outcomes = np.exp(mo.predict(
# 	    theta_fixed,
# 	    outcomes[:, None], 
# 	    ipe_samps[orders[i]][t],
# 	    f_smooth))[:, 1]
# 	resp = np.random.rand() < p_outcomes
# 	p_responses[1, :, i, t] = (resp * sd[t]) + ((1-resp) * (1-sd[t]))

# 	# fixed at uniform belief
# 	p_outcomes = np.exp(mo.predict(
# 	    theta_uniform[None],
# 	    outcomes[:, None], 
# 	    ipe_samps[orders[i]][t],
# 	    f_smooth))[:, 1]
# 	resp = np.random.rand() < p_outcomes
# 	p_responses[2, :, i, t] = (resp * sd[t]) + ((1-resp) * (1-sd[t]))


# <codecell>


