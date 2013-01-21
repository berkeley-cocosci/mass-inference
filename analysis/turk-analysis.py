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

pd.set_option('line_width', 195)
LINE = "-"*195

# <codecell>

def print_data(name, data):
    print name
    print LINE
    print data.to_string()
    print
	

# <codecell>

# global variables
normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))

# <codecell>

def make_truth_df(rawtruth, rawsstim, kappas, nthresh0):
    nfell = (rawtruth['nfellA'] + rawtruth['nfellB']) / 10.0
    truth = nfell > nthresh0
    truth[np.isnan(nfell)] = 0.5
    df = pd.DataFrame(truth[..., 0].T, index=kappas, columns=rawsstim)
    return df

def make_ipe_df(rawipe, rawsstim, kappas, nthresh):
    nfell = (rawipe['nfellA'] + rawipe['nfellB']) / 10.0
    samps = (nfell > nthresh).astype('f8')
    # samps = nfell.copy()
    samps[np.isnan(nfell)] = 0.5
    alpha = np.sum(samps, axis=-1) + 0.5
    beta = np.sum(1-samps, axis=-1) + 0.5
    pfell_mean = alpha / (alpha + beta)
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    pfell_meanstd = np.mean(pfell_std, axis=-1)
    ipe = np.empty(pfell_mean.shape)
    for idx in xrange(rawipe.shape[0]):
        x = kappas
        lam = pfell_meanstd[idx] * 10
        kde_smoother = mo.make_kde_smoother(x, lam)
        ipe[idx] = kde_smoother(pfell_mean[idx])
    df = pd.DataFrame(ipe.T, index=kappas, columns=rawsstim)
    return df
		  

# <codecell>

def plot_smoothing(rawipe, stims, nstim, nthresh):
    nfell = (rawipe['nfellA'] + rawipe['nfellB']) / 10.0
    samps = (nfell > nthresh).astype('f8')
    # samps = nfell.copy()
    samps[np.isnan(nfell)] = 0.5
    # samps = np.concatenate([
    #     ((rawipe['nfellA'] + rawipe['nfellB']) > nthresh).astype(
    #         'int')[..., None]], axis=-1)[..., 0][..., :nsamps]
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

    # plt.figure(fignum)
    plt.figure()
    plt.clf()
    plt.suptitle(
        "Likelihood function for feedback given mass ratio\n"
        "(%d IPE samples, threshold=%d%% blocks)" % (rawipe0.shape[2], nthresh*100),
        fontsize=16)
    plt.ylim(0, 1)
    plt.xticks(xticks, xticks10)
    plt.xlabel("Mass ratio ($r$)", fontsize=14)
    plt.yticks(yticks, yticks)
    plt.ylabel("\Pr(fall|$r$, $S$)", fontsize=14)
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
                     label=str(stims[i]).split("_")[1])
    # plt.legend(loc=8, prop={'size':12}, numpoints=1,
    #            scatterpoints=1, ncol=3, title="Stimuli")

# <codecell>

def plot_belief(model_theta):
    r, c = 3, 3
    n = r*c
    exp = np.exp(np.log(0.5) / np.log(1./27))    
    # fig = plt.figure(fignum)
    fig = plt.figure()
    plt.clf()
    gs = gridspec.GridSpec(r, c+1, width_ratios=[1]*c + [0.1])
    plt.suptitle(
        "Posterior belief about mass ratio over time",
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
        subjname = "True $r=%s$" % float(ratios[kidx])
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
            plt.ylabel("Mass ratio ($r$)", fontsize=14)
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
    cax.set_title("$\Pr(r|B_t)$", fontsize=14)
    

# <codecell>

########################################################################
def compute_ll(resp, theta, order):
    ll = np.empty(resp.shape)

    for t in order:

	thetas_t = theta[t][None]
	samps_t = ipe_samps[t]
	
	# compute likelihood of outcomes
	p_outcomes = np.exp(mo.predict(
	    thetas_t, outcomes[:, None], samps_t, f_smooth)).ravel()
	
	# observe response
	p_response = (resp[:, t]*p_outcomes[1]) + ((1-resp[:, t])*p_outcomes[0])

	ll[:, t] = np.log(p_response)

    return ll

# <codecell>

######################################################################
## Load human data
######################################################################

reload(lat)

training = {
    'A-fb': lat.load_turk_df("A-fb", "training"),
    'A-nfb': lat.load_turk_df("A-nfb", "training")
    }

posttest = {
    'A-fb': lat.load_turk_df("A-fb", "posttest"),
    'A-nfb': lat.load_turk_df("A-nfb", "posttest")
    }

experiment = {
    'A-fb': lat.load_turk_df("A-fb", "experiment"),
    'A-nfb': lat.load_turk_df("A-nfb", "experiment")
    }

# <codecell>

######################################################################
## Load model data
######################################################################

nthresh0 = 0
nthresh = 0.4

reload(lat)
rawtruth0, rawipe0, rawsstim, kappas = lat.load_model("stability")
truth0 = make_truth_df(rawtruth0, rawsstim, kappas, nthresh0)
ipe0 = make_ipe_df(rawipe0, rawsstim, kappas, nthresh)

hstim = np.array([x.split("~")[0] for x in zip(*experiment['A-fb'].columns)[1]])
sstim = np.array(ipe0.columns)
idx = np.nonzero((sstim[:, None] == hstim[None, :]))[0]

truth = truth0.T.ix[idx].T
ipe = ipe0.T.ix[idx].T
feedback = np.asarray(truth).T[..., None]
nfell = (rawipe0[idx]['nfellA'] + rawipe0[idx]['nfellB']) / 10.0
ipe_samps = (nfell > nthresh)[..., None].astype('f8')
# ipe_samps = (nfell / 10.0)[..., None]
ipe_samps[np.isnan(nfell)] = 0.5

plt.close('all')
plot_smoothing(rawipe0[idx], hstim, 6, nthresh=nthresh)
fig = plt.gcf()
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
responses    = np.array([0, 1])                  # possible responses
n_trial      = hstim.size
n_kappas     = len(kappas)                       # number of mass ratios to consider
n_responses  = responses.size                    # number of possible responses
n_outcomes   = outcomes.size                     # number of possible outcomes
kappa0       = 1.0                               # value of the true log mass ratio
ikappa0      = np.nonzero(kappas==1.0)[0][0]     # index for the true mass ratio

f_smooth = False
p_ignore_stimulus = 0.0

# <codecell>

######################################################################
## Generate fake human data
######################################################################
nfake = 1000

# for cidx, cond in enumerate(experiment.keys()):
#     if cond.endswith("-mo"): continue
#     if cond.endswith("-nfb"): continue

cond = 'A-fb'

# trial ordering
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

# <codecell>

plt.close('all')
plot_belief(model_theta)
fig = plt.gcf()
fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/ideal_learning_observers.png", close=False)

# <codecell>

# ideal observer model (k=0.1)
tidx = list(kappas).index(-1.0)
p_outcomes = np.empty((n_trial,))
for t in xrange(n_trial):
    p_outcomes[t] = np.exp(mo.predict(
	model_theta[tidx, t][None],
	outcomes[:, None], 
	ipe_samps[order][t],
	f_smooth)).ravel()[1]
responses = np.random.rand(nfake)[:, None] < p_outcomes[None]		
experiment[cond + "-ideal0.1-mo"] = pd.DataFrame(
    responses[:, undo_order], 
    columns=cols)

# <codecell>

# ideal observer model (k=10)
p_outcomes = np.empty((n_trial,))
for t in xrange(n_trial):
    p_outcomes[t] = np.exp(mo.predict(
	model_theta[list(kappas).index(1.0), t][None],
	outcomes[:, None], 
	ipe_samps[order][t],
	f_smooth)).ravel()[1]
responses = np.random.rand(nfake)[:, None] < p_outcomes[None]		
experiment[cond + "-ideal10-mo"] = pd.DataFrame(
    responses[:, undo_order], 
    columns=cols)

# <codecell>

# fixed r=0.1 model
p_outcomes = np.empty((n_trial,))
for t in xrange(n_trial):
    theta = np.log(np.zeros(n_kappas))
    theta[list(kappas).index(-1.0)] = 0
    theta = normalize(theta)[1]
    p_outcomes[t] = np.exp(mo.predict(
	theta[None],
	outcomes[:, None], 
	ipe_samps[order][t],
	f_smooth)).ravel()[1]
responses = np.random.rand(nfake)[:, None] < p_outcomes[None]		
experiment[cond + "-fixed0.1-mo"] = pd.DataFrame(
    responses[:, undo_order], 
    columns=cols)

# <codecell>

# fixed r=10.0 model
p_outcomes = np.empty((n_trial,))
for t in xrange(n_trial):
    theta = np.log(np.zeros(n_kappas))
    theta[list(ratios).index(10.0)] = 0
    theta = normalize(theta)[1]
    p_outcomes[t] = np.exp(mo.predict(
	theta[None],
	outcomes[:, None], 
	ipe_samps[order][t],
	f_smooth)).ravel()[1]
responses = np.random.rand(nfake)[:, None] < p_outcomes[None]		
experiment[cond + "-fixed10-mo"] = pd.DataFrame(
    responses[:, undo_order], 
    columns=cols)

# <codecell>

# conds = sorted(experiment.keys())
conds = [
    'A-fb-fixed0.1-mo',
    'A-fb-ideal0.1-mo', 
    'A-fb-fixed10-mo', 
    'A-fb-ideal10-mo',
    'A-nfb', 
    'A-fb',
    ]
cond_labels = {
    'A-nfb': 'No feedback',
    'A-fb': 'Feedback',
    'A-fb-ideal0.1-mo': 'Learning observer, r=0.1',
    'A-fb-ideal10-mo': 'Learning observer, r=10',
    'A-fb-fixed0.1-mo': 'Fixed Observer, r=0.1',
    'A-fb-fixed10-mo': 'Fixed Observer, r=10'
    }
n_cond = len(conds)
    

# <codecell>

# for cond in training:
#     print_data("%s TRAINING DATA" % cond, training[cond])
# for cond in posttest:
#     print_data("%s POSTTEST DATA" % cond, posttest[cond])

# for cond in conds:
#     if cond.endswith("mo"): continue
#     print_data("%s EXPERIMENT DATA" % cond, experiment[cond])

# <codecell>

# raw correlation between feedback and no feedback
means = [experiment[cond].mean(axis=0) for cond in conds]
for idx1, m1 in enumerate(means):
    for idx2, m2 in enumerate(means[idx1+1:]):
	corr = xcorr(np.asarray(m1), np.asarray(m2))
	c1 = conds[idx1]
	c2 = conds[idx1+idx2+1]
	print "(all) %-16s v %-16s : rho = % .4f" % (c1, c2, corr)
	

# <codecell>

# bootstrapped correlations
reload(lat)

nboot = 1000
nsamp = 9
with_replacement = False

A_fb = np.asarray(experiment['A-fb'])
A_nfb = np.asarray(experiment['A-nfb'])

corrs = lat.bootcorr(
    np.asarray(A_fb), np.asarray(A_nfb), 
    nboot=nboot, 
    nsamp=nsamp,
    with_replacement=with_replacement)
meancorr = np.median(corrs)
semcorr = scipy.stats.sem(corrs)
print "(bootstrap) feedback v no-feedback: rho = %.4f +/- %.4f" % (meancorr, semcorr)

corrs = lat.bootcorr_wc(
    np.asarray(A_fb), 
    nboot=nboot, 
    nsamp=nsamp,
    with_replacement=with_replacement)
meancorr = np.median(corrs)
semcorr = scipy.stats.sem(corrs)
print "(bootstrap) feedback v feedback: rho = %.4f +/- %.4f" % (meancorr, semcorr)

corrs = lat.bootcorr_wc(
    np.asarray(A_nfb), 
    nboot=nboot,
    nsamp=nsamp,
    with_replacement=with_replacement)
meancorr = np.median(corrs)
semcorr = scipy.stats.sem(corrs)
print "(bootstrap) no-feedback v no-feedback: rho = %.4f +/- %.4f" % (meancorr, semcorr)

# <codecell>


p_response_mean = np.empty((n_kappas, n_cond))
p_response_sem = np.empty((n_kappas, n_cond))

for cidx, cond in enumerate(conds):
    hdata = np.asarray(experiment[cond])[:, None]
    sdata = np.asarray(ipe)[None]
    
    resp = (hdata * sdata) + ((1-hdata) * (1-sdata))
    p_response_mean[:, cidx] = np.mean(np.exp(np.log(resp).sum(axis=2)), axis=0)
    p_response_sem[:, cidx] = scipy.stats.sem(np.exp(np.log(resp).sum(axis=2)), axis=0)

x = np.arange(n_kappas)
upper = np.log(p_response_mean + p_response_sem).T
lower = np.log(p_response_mean - p_response_sem).T
mean = np.log(p_response_mean).T

plt.close('all')
colors = ['r', '#FF9966', '#AAAA00', 'g', 'c', 'b', '#9900FF', 'm']
for cidx, cond in enumerate(conds):
    plt.fill_between(x, lower[cidx], upper[cidx], color=colors[cidx], alpha=0.1)
    plt.plot(x, mean[cidx], label=cond_labels[cond], color=colors[cidx], linewidth=2)

plt.xticks(x, ratios, rotation=90)
plt.xlabel("Fixed model mass ratio")
plt.ylabel("Log likelihood of responses")
plt.legend(loc=4)
plt.xlim(x[0], x[-1])
#plt.ylim(-20, -12)
plt.title("Likelihood of responses under fixed models")
fig = plt.gcf()
fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/fixed_model_performance.png", close=False)

# <codecell>

# fixed models
kidx = list(kappas).index(0.0)
model_same = np.array((
    mean[:, kidx],
    np.abs(mean[:, kidx] - lower[:, kidx]),
    np.abs(mean[:, kidx] - upper[:, kidx]))).T

kidx = list(kappas).index(1.0)
model_true10 = np.array((
    mean[:, kidx],
    np.abs(mean[:, kidx] - lower[:, kidx]),
    np.abs(mean[:, kidx] - upper[:, kidx]))).T

kidx = list(kappas).index(-1.0)
model_true01 = np.array((
    mean[:, kidx],
    np.abs(mean[:, kidx] - lower[:, kidx]),
    np.abs(mean[:, kidx] - upper[:, kidx]))).T

# random model
model_random = np.array([[np.log(0.5)*n_trial, 0, 0]]*n_cond)

# <codecell>


def learning_model_lh(ikappa):
    lh_mean = np.empty(n_cond)
    lh_sem = np.empty(n_cond)
    
    for cidx, cond in enumerate(conds):
	# trial ordering
	order = np.argsort(zip(*experiment[cond].columns)[0])
	# learning model beliefs
	model_lh, model_joint, model_theta = mo.ModelObserver(
	    ipe_samps[order],
	    feedback[order][:, None],
	    outcomes=None,
	    respond=False,
	    p_ignore_stimulus=p_ignore_stimulus,
	    smooth=f_smooth)

	# trial-by-trial likelihoods of judgments
	trial_ll = compute_ll(
	    np.asarray(experiment[cond]), 
	    model_theta[ikappa], 
	    order)

	# overall likelihood
	lh = np.exp(np.sum(trial_ll, axis=1))

	# mean across participants
	lh_mean[cidx] = np.mean(lh)
	lh_sem[cidx] = scipy.stats.sem(lh)

    mean = np.log(lh_mean)
    lower = np.log(lh_mean - lh_sem)
    upper = np.log(lh_mean + lh_sem)

    return mean, lower, upper

# <codecell>

# learning models
mean, lower, upper = learning_model_lh(list(kappas).index(1.0))
model_learn10 = np.array([mean, np.abs(mean-lower), np.abs(mean-upper)]).T

mean, lower, upper = learning_model_lh(list(kappas).index(-1.0))
model_learn01 = np.array([mean, np.abs(mean-lower), np.abs(mean-upper)]).T

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

# plot model performance
x0 = np.arange(models.shape[2])
height = models[0]
err = models[1:]
width = 0.7 / n_cond

plt.close('all')
for cidx, cond in enumerate(conds):
    x = x0 + width*(cidx-(n_cond/2.)) + (width/2.)
    plt.bar(x, height[cidx], yerr=err[:, cidx], color=colors[cidx], 
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
plt.legend(loc=0)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Log likelihood of responses, $\Pr(J|S,B)$", fontsize=14)
plt.title("Likelihood of human and ideal observer judgments", fontsize=16)

fig = plt.gcf()
fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/model_performance.png", close=False)

# <codecell>


nsamp = 1000
ikappa = list(kappas).index(1.0)
orders = np.array([np.random.permutation(n_trial) for i in xrange(nsamp)])

# <codecell>

model_thetas = np.empty((nsamp, n_kappas, n_trial+1, n_kappas))
for i in xrange(model_thetas.shape[0]):
    if i%10 == 0:
	print i
    model_lh, model_joint, model_theta = mo.ModelObserver(
	ipe_samps[orders[i]],
	feedback[orders[i]][:, None],
	outcomes=None,
	respond=False,
	p_ignore_stimulus=p_ignore_stimulus,
	smooth=f_smooth)
    model_thetas[i] = model_theta

# <codecell>


p_responses = np.empty((3, n_kappas, nsamp, n_trial))

theta_fixed = np.log(np.eye(n_kappas))

theta_uniform = normalize(np.log(np.ones(n_kappas)))[1]
sdata = np.asarray(ipe)

for i in xrange(nsamp):
    if i%10 == 0:
	print i
    sd = sdata[ikappa, orders[i]]
    for t in xrange(n_trial):
	# learning
	p_outcomes = np.exp(mo.predict(
	    model_thetas[i, :, t],
	    outcomes[:, None], 
	    ipe_samps[orders[i]][t],
	    f_smooth))[:, 1]
	resp = np.random.rand() < p_outcomes
	p_responses[0, :, i, t] = (resp * sd[t]) + ((1-resp) * (1-sd[t]))

	# fixed at true ratio
	p_outcomes = np.exp(mo.predict(
	    theta_fixed,
	    outcomes[:, None], 
	    ipe_samps[orders[i]][t],
	    f_smooth))[:, 1]
	resp = np.random.rand() < p_outcomes
	p_responses[1, :, i, t] = (resp * sd[t]) + ((1-resp) * (1-sd[t]))

	# fixed at uniform belief
	p_outcomes1 = np.exp(mo.predict(
	    theta_uniform[None],
	    outcomes[:, None], 
	    ipe_samps[orders[i]][t],
	    f_smooth))[:, 1]
	resp = np.random.rand() < p_outcomes
	p_responses[2, :, i, t] = (resp * sd[t]) + ((1-resp) * (1-sd[t]))


# <codecell>

p_response_mean = np.mean(p_responses, axis=2)
p_response_sem = scipy.stats.sem(p_responses, axis=2)

x = np.arange(n_trial)
upper = p_response_mean + p_response_sem
lower = p_response_mean - p_response_sem
mean = p_response_mean

plt.close('all')
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

lat.save("images/likelihoods_over_time.png", close=False)

# <codecell>


