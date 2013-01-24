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

training = {}
posttest = {}
experiment = {}
queries = {}
    

# <codecell>

######################################################################
## Check human data for bad participants
######################################################################

dfs = []
suffix = ['-cb0', '-cb1']
for c in ['B-fb-10', 'B-fb-0.1', 'B-nfb-10']:
    for s in suffix:
	cond = c + s
	dfs.append(lat.load_turk_df(cond, "posttest"))

for cond in posttest.keys():
    dfs.append(posttest[cond])
df = pd.concat(dfs)
mean = df.mean(axis=0)
print
print mean
wrong = (df != mean.round()).sum(axis=1)
bad = wrong > 1
pids = sorted(list((bad).index[(bad).nonzero()]))
print
print "Bad pids: ", pids

# <codecell>

######################################################################
## Load human data
######################################################################

reload(lat)
suffix = ['-cb0', '-cb1']
for cond in ['B-fb-10', 'B-fb-0.1', 'B-nfb-10']:
    conds = [cond+s for s in suffix]
    for c in conds:
	training[c] = lat.load_turk_df(c, "training", exclude=pids)
	posttest[c] = lat.load_turk_df(c, "posttest", exclude=pids)
	experiment[c] = lat.load_turk_df(c, "experiment", exclude=pids)
	if cond.split("-")[1] != "nfb":
	    queries[c] = lat.load_turk_df(c, "queries", exclude=pids)

# <codecell>

######################################################################
## Load model data
######################################################################

nthresh0 = 0
nthresh = 0.4

reload(lat)
rawtruth0, rawipe0, rawsstim, kappas = lat.load_model("stability")
truth0 = lat.make_truth_df(rawtruth0, rawsstim, kappas, nthresh0)
ipe0 = lat.make_ipe_df(rawipe0, rawsstim, kappas, nthresh)

hstim = np.array([x.split("~")[0] for x in zip(*experiment['B-fb-10-cb0'].columns)[1]])
sstim = np.array(ipe0.columns)
idx = np.nonzero((sstim[:, None] == hstim[None, :]))[1]

truth = truth0.T.ix[idx].T
ipe = ipe0.T.ix[idx].T
feedback = np.asarray(truth).T[..., None]
nfell = (rawipe0[idx]['nfellA'] + rawipe0[idx]['nfellB']) / 10.0
ipe_samps = (nfell > nthresh)[..., None].astype('f8')
ipe_samps[np.isnan(nfell)] = 0.5

fig = plt.figure(1)
plt.clf()
lat.plot_smoothing(rawipe0[idx], hstim, 6, nthresh, kappas)
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

outcomes     = np.array([0, 1])                  # possible outcomes
n_trial      = hstim.size
n_outcomes   = outcomes.size                     # number of possible outcomes

f_smooth = True
p_ignore_stimulus = 0.0

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))

# <codecell>

######################################################################
## Generate fake human data
######################################################################
nfake = 50
for cond in experiment.keys():

    if cond.startswith("MO-"):
	continue

    cols = experiment[cond].columns
    order = np.argsort(zip(*cols)[0])
    undo_order = np.argsort(order)
    
    # learning model beliefs
    model_lh, model_joint, model_theta = mo.ModelObserver(
	ipe_samps[order],
	feedback[order][:, None],
	outcomes=None,
	respond=False,
	p_ignore_stimulus=p_ignore_stimulus,
	smooth=f_smooth)
    
    p_outcomes = np.empty((n_trial,))
    args = cond.split("-")
    if len(args) > 2:
	tidx = list(ratios).index(float(args[2]))
    else:
	tidx = None
	
    for t in xrange(n_trial):
	if args[1] == "nfb":
	    thetas = [
		np.log(np.zeros(n_kappas)),
		normalize(np.log(np.ones(n_kappas)))[1]]
	    thetas[0][list(kappas).index(0.0)] = 0
	    newconds = [
		"-".join(["MO-nfb-1.0"] + args[3:]),
		"-".join(["MO-nfb"] + args[3:])]
		
	elif args[1] == "fb":
	    thetas = [
		np.log(np.zeros(n_kappas)),
		model_theta[tidx, t]
		]
	    thetas[0][tidx] = 0
	    newconds = ["-".join(["MO-nfb"] + args[2:]),
			"-".join(["MO-fb"] + args[2:])]

	for theta, newcond in zip(thetas, newconds):
	    p_outcomes[t] = np.exp(mo.predict(
		    theta[None],
		    outcomes[:, None], 
		    ipe_samps[order][t],
		    f_smooth)).ravel()[1]
	    responses = np.random.rand(nfake)[:, None] < p_outcomes[None]		
	    experiment[newcond] = pd.DataFrame(
		    responses[:, undo_order], 
		    columns=cols)

# <codecell>

reload(lat)
fig = plt.figure(2)
plt.clf()
lat.plot_belief(model_theta, kappas, cmap)
fig.set_figwidth(8)
fig.set_figheight(2.5)

lat.save("images/ideal_learning_observers.png", close=False)

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
    'MO-nfb': 'Uniform fixed observer',
    'MO-nfb-1.0': '(r=1) Fixed observer',
    'MO-fb-0.1': '(r=0.1) Learning observer',
    'MO-fb-10': '(r=10) Learning observer',
    'MO-nfb-0.1': '(r=0.1) Fixed observer',
    'MO-nfb-10': '(r=10) Fixed observer',
    }

conds = sorted(experiment.keys())
condsort = np.argsort(cond_labels.values())
newconds = list([str(x) for x in np.array(cond_labels.keys())[condsort]])
n_cond = len(newconds)

# cond_labels = dict([(c, c) for c in conds])
# newconds = sorted(np.unique(["-".join(c.split("-")[:-1]) for c in conds]))
    

# <codecell>

# bootstrapped correlations
reload(lat)

nboot = 1000
nsamp = 9
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
    meancorr = np.median(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond+"-cb0", cond+"-cb1", meancorr, semcorr)

    corrs = lat.bootcorr_wc(
	np.asarray(arr3), 
	nboot=nboot,
	nsamp=nsamp,
	with_replacement=with_replacement)
    meancorr = np.median(corrs)
    semcorr = scipy.stats.sem(corrs)
    print "(bootstrap) %-15s v %-15s: rho = %.4f +/- %.4f" % (
	cond, cond, meancorr, semcorr)

# <codecell>

def collapse(data):
    stacked = [np.vstack([data[c] for c in conds if c.startswith(nc)]) for nc in newconds]
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

colors = ['r', '#FF9966', '#AAAA00', 'g', 'c', 'b', '#9900FF', 'm']
for cidx, cond in enumerate(newconds):
    color = colors[cidx % len(colors)]
    if cidx >= len(colors):
	linestyle = '--'
    else:
	linestyle = '-'
    plt.fill_between(x, lower[cidx], upper[cidx], color=color, alpha=0.1)
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
window = 8
lh = np.empty((n_trial-window, len(newconds), 3))
x = np.arange(n_trial-window)
for t in xrange(n_trial-window):
    lh[t] = collapse(lat.fixed_model_lh(
	conds, experiment, ipe_samps, thetas, t, t+window, f_smooth))[0]

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
		     color=color, alpha=0.1)
    plt.plot(lh[:, cidx, 0], color=color, label=cond, linestyle=linestyle)

	
plt.legend(loc=0, ncol=3)
fig.set_figwidth(8)
fig.set_figheight(6)
plt.xlim(0, n_trial-window-1)
plt.xlabel("Trial")
plt.ylabel("Likelihood")
plt.title("Likelihood of observer responses, averaged over trial orderings")
plt.ylim(-7.5, -3.5)

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
    plt.fill_between(x, lower[i, j], upper[i, j], color=colors[k], alpha=0.1)
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


