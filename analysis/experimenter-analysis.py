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

# nicknames
normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample
rso = np.random.RandomState(0)

# <codecell>

# load stability data
out = lat.load('stability')
rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = out

# <codecell>

# order data by trial
human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, rawtruth, rawipe)
truth = truth[0]
ipe = ipe[0]
print "truth", truth.shape
print "ipe", ipe.shape

# <codecell>

# human responses
ifell = human > 3
istable = human < 3
ichance = human == 3
h_responses = np.empty(human.shape)
h_responses[ifell] = 1
h_responses[istable] = 0
h_responses[ichance] = rso.randint(0, 2, h_responses[ichance].shape)

plt.figure()
plt.bar(np.arange(h_responses.shape[0]),
        np.mean(h_responses, axis=1), align='center')
plt.title("Probability of human responding \"fall\"")
plt.ylim(0, 1)
plt.xlim(-1, 11)

# <codecell>

# compute ratios (from kappa values)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)
print ratios.shape; ratios
 

# <codecell>

# plotting parameters
cmap = lat.make_cmap(                            # colormap for plotting
    "lh", 
    (0, 0, 0),    # black for low values
    (.5, .5, .5), # grey for mid values
    (1, 0, 0))    # red for high values

# global parameters
outcomes     = np.array([0, 1])                  # possible outcomes
responses    = np.array([0, 1])                  # possible responses
n_trial      = stimuli.shape[1]                  # number of trials
n_kappas     = len(kappas)                       # number of mass ratios to consider
n_responses  = responses.size                    # number of possible responses
n_outcomes   = outcomes.size                     # number of possible outcomes
kappa0       = 1.0                               # value of the true log mass ratio
ikappa0      = np.nonzero(kappas==1.0)[0][0]     # index for the true mass ratio

# model observer parameters
nthresh0          = 1                            # feedback threshold for "fall"
nthresh           = 4                            # ipe threshold for "fall"
nsamps            = 300                          # number of ipe samples
f_smooth          = True                         # smooth ipe likelihoods
p_ignore_stimulus = 0.0                          # probability of ignoring stimulus

# particle filter parameters
n_part       = 200                               # number of particles
neffthresh   = n_part * 0.75                     # number of effective particles threshold
kw           = 0.1                               # kernel width for smoothing
pn           = 0.1                               # noise
# B            = 1                                 # 
# rand         = 0.5                               # 

def make_smoother(kw):
    x = np.arange(0, n_kappas*0.1, 0.1)
    dists = np.abs(x[:, None] - x[None, :]) / kw
    pdist = np.exp(-0.5 * dists**2) / np.sqrt(2)
    sum_pdist = np.sum(pdist, axis=-1)
    def smoother(v):
        return np.sum(pdist * v, axis=-1) / sum_pdist
    return smoother

smoother = make_smoother(kw)

# <codecell>

# get feedback and IPE model data
feedback, ipe_samps = lat.make_observer_data(
    nthresh0, nthresh, nsamps)
print "feedback", feedback.shape
print "ipe     ", ipe_samps.shape

# <codecell>

# compute belief for model observer
m_lh, m_joint, m_kappas, m_responses = mo.ModelObserver(
    ipe_samps,
    feedback[:, None],
    outcomes=outcomes[:, None],
    respond=True,
    p_ignore_stimulus=p_ignore_stimulus,
    smooth=f_smooth)

print "P_t(k)", m_kappas.shape; m_kappas

exp = np.exp(np.log(0.5) / np.log(1. / n_kappas))
logcticks = np.array([0, 0.001, 0.05, 0.25, 1])
cticks = np.exp(np.log(logcticks) * np.log(exp))

plt.figure()
plt.imshow(exp**normalize(m_kappas[:, -1].T, axis=0)[1], cmap=cmap, interpolation='nearest', vmin=0, vmax=1, origin='lower')
plt.xlabel("Feedback condition")
plt.ylabel("Mass ratio")
plt.title("Model observer belief as a function of feedback")
cb = plt.colorbar(ticks=cticks)
cb.set_ticklabels(logcticks)

# <codecell>

plt.figure()
plt.plot(kappas, np.mean(m_responses, axis=1))
plt.xticks(plt.xticks()[0], np.round(10**plt.xticks()[0], decimals=2))
plt.title("Probability of model responding \"fall\"")
plt.xlabel("Feedback condition")
plt.ylim(0, 1)

# <codecell>


# "true" responses (for the purposes of inference)

true_responses = m_responses.copy().astype('i8')
# true_responses = mo.responses(
#     m_kappas[:, [-1]]*np.ones(m_kappas.shape),
#     outcomes[:, None], ipe_samps, loss, f_smooth)
#sidxs = [0, 3, 6, 8, 10, 13, 16, 18, 20, 23, 26]
sidxs = [0, 13, 26]

# true_responses = h_responses.copy()
# sidxs = np.arange(true_responses.shape[0])

n_subj = len(sidxs)
print "Running inference on %d subjects" % n_subj

# <codecell>

##########################
##### INITIALIZATION #####
##########################

# arrays to hold particles and weights
thetas = np.empty((n_subj, n_part, n_trial+1, n_kappas)) * np.nan
weights = np.empty((n_subj, n_part, n_trial+1)) * np.nan
mle_alphas = np.empty((n_subj, n_trial+1, n_kappas)) * np.nan

rso = np.random.RandomState(50)
def make_prior(alpha=5, ell=0.2, eps=0.0005):
    kernel = mo.make_rbf_kernel(alpha=alpha, ell=ell)
    K = kernel(kappas[:, None], kappas[None, :]) + (eps * np.eye(n_kappas))
    def draw(n):
        f = rso.multivariate_normal(np.zeros(n_kappas), K, n)
        sf = normalize(np.log(1. / (1. + np.exp(-f))), axis=-1)[1]
        return sf
    return draw

thetas[:, :, 0] = np.log(np.random.dirichlet(np.ones(n_kappas), (n_subj, n_part)))
#thetas[:, :, 0] = sf
#thetas[:, :, 0] = np.log(1. / n_kappas)
weights[:, :, 0] = np.log(np.ones((n_subj, n_part)) / n_part)
mle_alphas[:, 0] = np.ones((n_subj, n_kappas)) / n_kappas


# <codecell>

###########################
##### PARTICLE FILTER #####
###########################

for idx, sidx in enumerate(sidxs):

    print idx
    rso.seed(100)
        
    for t in xrange(0, n_trial):

        thetas_t = thetas[idx, :, t].copy()
        weights_t = weights[idx, :, t].copy()
        samps_t = ipe_samps[t]
        #obs_t = feedback[t, sidx]
        obs_t = feedback[t, ikappa0]
        response_t = true_responses[sidx, t]

        if not np.isnan(response_t):
            response_t = int(response_t)
             
            # propagate particles
            ef = mo.evaluateFeedback(obs_t, samps_t, f_smooth)
            noise = np.random.dirichlet(np.ones(n_kappas), (n_part,))
            #noise = np.zeros(thetas_t.shape)
            v = np.exp(normalize(np.log((1-pn)*np.exp(thetas_t + ef) + pn*noise), axis=-1)[1])
            m_theta = normalize(np.log(smoother(v[:, None])), axis=-1)[1]
            #m_theta = normalize(thetas_t + ef + noise, axis=-1)[1]

            # compute new weights
            p_outcomes = np.exp(mo.predict(
                thetas_t, outcomes[:, None], samps_t, f_smooth))
            p_response = (p_outcomes[:, response_t]*(1-p_ignore_stimulus) + 
                          (1./n_responses)*(p_ignore_stimulus))
            weights_t = normalize(np.log(p_response) + weights_t)[1]
            
            # compute number of effective particles
            neff = 1. / np.sum(np.exp(weights_t)**2)
            #print neff
            if neff < neffthresh:
                # sample new particles
                tidx = weightedSample(
                    np.exp(weights_t), n_part, rso=rso)

                # update
                thetas[idx, :, t+1] = m_theta[tidx]
                weights[idx, :, t+1] = np.log(
                    np.ones(n_part, dtype='f8') / n_part)
            else:
                thetas[idx, :, t+1] = m_theta
                weights[idx, :, t+1] = weights_t

        else:
            thetas[idx, :, t+1] = thetas_t
            weights[idx, :, t+1] = weights_t

# <codecell>

fig = plt.figure(3)
plt.clf()
fig.set_figwidth(9)
fig.set_figheight(3)
r, c = 1, 3
n = r*c
exp = np.exp(np.log(0.5) / np.log(1./27))    
gs = gridspec.GridSpec(r, c+1, width_ratios=[1]*c + [0.1])
plt.suptitle("MLE P(kappa)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)

for i, sidx in enumerate(sidxs):
    irow, icol = np.unravel_index(i, (r, c))
    ax = plt.subplot(gs[irow, icol])
    kappa = kappas[sidx]
    subjname = "Observed $\kappa=%s$" % float(ratios[sidx])
    img = lat.plot_theta(
        None, None, ax,
        np.mean(np.exp(thetas[i]), axis=0),
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

    # subjname = "Model Subj. r=%.1f" % ratios[sidx]
    # print subjname
    # mle_theta = np.mean(np.exp(thetas[sidx]), axis=0)
    # plt.figure(2)
    # lat.plot_theta(
    #     3, 4, sidx+1,
    #     mle_theta, subjname, exp=np.e, ratios=ratios)

# <codecell>


