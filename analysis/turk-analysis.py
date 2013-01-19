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

nthresh0 = 1
nthresh = 4

# <codecell>

def make_truth_df(rawtruth, rawsstim, kappas, nthresh0):
    truth = (rawtruth['nfellA'] + rawtruth['nfellB']) > nthresh0
    df = pd.DataFrame(truth[..., 0].T, index=kappas, columns=rawsstim)
    return df

def make_ipe_df(rawipe, rawsstim, kappas, nthresh):
    samps = (rawipe['nfellA'] + rawipe['nfellB']) > nthresh
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

######################################################################
## Load model data
reload(lat)
rawtruth0, rawipe0, rawsstim, kappas = lat.load_model("stability")

truth0 = make_truth_df(rawtruth0, rawsstim, kappas, nthresh0)
ipe0 = make_ipe_df(rawipe0, rawsstim, kappas, nthresh)

n_kappas = len(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# <codecell>

######################################################################
## Load human data
reload(lat)

A_fb_training = lat.load_turk_df("A-fb", "training")
A_fb_posttest = lat.load_turk_df("A-fb", "posttest")
A_fb = lat.load_turk_df("A-fb", "experiment")

A_nfb_training = lat.load_turk_df("A-nfb", "training")
A_nfb_posttest = lat.load_turk_df("A-nfb", "posttest")
A_nfb = lat.load_turk_df("A-nfb", "experiment")

# <codecell>

print_data("A-fb TRAINING DATA", A_fb_training)
print_data("A-nfb TRAINING DATA", A_nfb_training)

# <codecell>

print_data("A-fb POSTTEST DATA", A_fb_posttest)
print_data("A-nfb POSTTEST DATA", A_nfb_posttest)

# <codecell>

print_data("A-fb EXPERIMENT DATA", A_fb)
print_data("A-nfb EXPERIMENT DATA", A_nfb)

# <codecell>

# raw correlation between feedback and no feedback
m1 = np.mean(np.asarray(A_fb), axis=0)
m2 = np.mean(np.asarray(A_nfb), axis=0)
corr = xcorr(m1, m2)
print "(all) feedback v no-feedback: rho = %.4f" % corr

# <codecell>

# bootstrapped correlations
reload(lat)

nboot = 10000
nsamp = 9
with_replacement = False

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

plt.close('all')

hstim = np.array([x.split("~")[0] for x in zip(*A_fb.columns)[1]])
sstim = np.array(ipe0.columns)

idx = np.nonzero((sstim[:, None] == hstim[None, :]))[0]
truth = truth0.T.ix[idx].T
ipe = ipe0.T.ix[idx].T

p_resp_fb = np.empty((A_fb.shape[0], n_kappas))
p_resp_nfb = np.empty((A_nfb.shape[0], n_kappas))
p_response_mean = np.empty((n_kappas, 2))
p_response_sem = np.empty((n_kappas, 2))

for kidx, kappa in enumerate(kappas):
    resp_fb = ((A_fb * np.asarray(ipe.ix[kappa])) + 
	       ((1-A_fb) * np.asarray(1-ipe.ix[kappa])))
    resp_nfb = ((A_nfb * np.asarray(ipe.ix[kappa])) + 
		((1-A_nfb) * np.asarray(1-ipe.ix[kappa])))

    p_resp_fb[:, kidx] = np.log(resp_fb).sum(axis=1)
    p_resp_nfb[:, kidx] = np.log(resp_nfb).sum(axis=1)

# p_resp_fb = normalize(p_resp_fb, axis=1)[1]
# p_resp_nfb = normalize(p_resp_nfb, axis=1)[1]
	
p_response_mean[:, 0] = np.exp(p_resp_fb).mean(axis=0)
p_response_mean[:, 1] = np.exp(p_resp_nfb).mean(axis=0)
p_response_sem[:, 0] = scipy.stats.sem(np.exp(p_resp_fb), axis=0)
p_response_sem[:, 1] = scipy.stats.sem(np.exp(p_resp_nfb), axis=0)

x = np.arange(n_kappas)
upper = np.log(p_response_mean + p_response_sem).T
lower = np.log(p_response_mean - p_response_sem).T
mean = np.log(p_response_mean).T

# for sidx in xrange(p_resp_fb.shape[0]):
#     plt.plot(x, p_resp_fb[sidx], color='#FFCCCC')
# for sidx in xrange(p_resp_nfb.shape[0]):
#     plt.plot(x, p_resp_nfb[sidx], color='#CCCCFF')

plt.fill_between(x, lower[0], upper[0], color='#FF0000', alpha=0.1)
plt.fill_between(x, lower[1], upper[1], color='#0000FF', alpha=0.1)
plt.plot(x, mean[0], label="feedback", color='#FF0000', linewidth=2)
plt.plot(x, mean[1], label="no feedback", color='#0000FF', linewidth=2)
plt.xticks(x, ratios, rotation=90)
plt.xlabel("Mass ratio")
plt.ylabel("Negative log likelihood")
plt.legend(loc=4)
plt.xlim(x[0], x[-1])
#plt.ylim(-15, 0)
plt.title("Likelihood of human responses under different models")

# <codecell>

model_same = np.array((mean[:, list(kappas).index(0.0)],
		       lower[:, list(kappas).index(0.0)],
		       upper[:, list(kappas).index(0.0)])).T
model_true = np.array((mean[:, list(kappas).index(1.0)],
		       lower[:, list(kappas).index(1.0)],
		       upper[:, list(kappas).index(1.0)])).T
model_opposite = np.array((mean[:, list(kappas).index(-1.0)],
			   lower[:, list(kappas).index(-1.0)],
			   upper[:, list(kappas).index(-1.0)])).T


# <codecell>

# random model
model_random = np.array([[np.log(0.5)*A_fb.shape[1]]*3,
			 [np.log(0.5)*A_nfb.shape[1]]*3])
model_random

# <codecell>

# global parameters
outcomes     = np.array([0, 1])                  # possible outcomes
responses    = np.array([0, 1])                  # possible responses
n_trial      = hstim.size
n_kappas     = len(kappas)                       # number of mass ratios to consider
n_responses  = responses.size                    # number of possible responses
n_outcomes   = outcomes.size                     # number of possible outcomes
kappa0       = 1.0                               # value of the true log mass ratio
ikappa0      = np.nonzero(kappas==1.0)[0][0]     # index for the true mass ratio

f_smooth = True
p_ignore_stimulus = 0.0

# <codecell>

hstim = np.array([x.split("~")[0] for x in zip(*A_fb.columns)[1]])
sstim = np.array(ipe0.columns)
idx = np.nonzero((sstim[:, None] == hstim[None, :]))[0]

feedback = np.asarray(truth).T[..., None]
ipe_samps = ((rawipe0[idx]['nfellA'] + rawipe0[idx]['nfellB']) > nthresh)[..., None]

# <codecell>

order_fb = np.argsort(zip(*A_fb.columns)[0])
order_nfb = np.argsort(zip(*A_nfb.columns)[0])

# <codecell>

model_lh, model_joint, model_theta = mo.ModelObserver(
    ipe_samps,
    feedback[:, None],
    outcomes=None,
    respond=False,
    p_ignore_stimulus=p_ignore_stimulus,
    smooth=f_smooth)

# <codecell>

def compute_ll(resp, order):
    ll = np.empty(resp.shape)

    for t in order:

	thetas_t = model_theta[ikappa0, t][None]
	samps_t = ipe_samps[t]
	
	# compute likelihood of outcomes
	p_outcomes = np.exp(mo.predict(
	    thetas_t, outcomes[:, None], samps_t, f_smooth)).ravel()
	
	# observe response
	p_response = (resp[:, t]*p_outcomes[1]) + ((1-resp[:, t])*p_outcomes[0])

	ll[:, t] = np.log(p_response)

    return ll

# <codecell>

lfb = np.exp(np.sum(compute_ll(np.asarray(A_fb), order_fb), axis=1))
lnfb = np.exp(np.sum(compute_ll(np.asarray(A_nfb), order_nfb), axis=1))

mean_p = np.array([np.mean(lfb), np.mean(lnfb)])
sem_p = np.array([scipy.stats.sem(lfb), scipy.stats.sem(lnfb)])

mean = np.log(mean_p)
lower = np.log(mean_p - sem_p)
upper = np.log(mean_p + sem_p)

model_learn = np.array([mean, lower, upper]).T

# <codecell>

models = np.array([model_random,
		   model_opposite,
		   model_same,
		   model_true,
		   model_learn]).T
height = models[0]
err = np.abs(height[None] - models[1:])

x = np.arange(models.shape[2])
xfb = x - 0.2
xnfb = x + 0.2

plt.close('all')
plt.bar(xfb, height[0], yerr=err[:, 0], color='#FF0000', ecolor='k', align='center', width=0.4, label="feedback")
plt.bar(xnfb, height[1], yerr=err[:, 1], color='#0000FF', ecolor='k', align='center', width=0.4, label="no feedback")
plt.xticks(x, ["random", "r=0.1", "r=1.0", "r=10.0", "learn"])
plt.ylim(-20, -12)
plt.legend(loc=0)
plt.xlabel("Model")
plt.ylabel("Negative log likelihood of responses")
plt.title("Model performance")

# <codecell>


