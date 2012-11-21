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

###############################
##### LOAD AND ORDER DATA #####
###############################

# load stability data
out = lat.load('stability')
rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = out

# order data by trial
human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, rawtruth, rawipe)
truth = truth[0]
ipe = ipe[0]
print "truth", truth.shape
print "ipe", ipe.shape

# <codecell>

# compute ratios (from kappa values)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)
print ratios.shape; ratios
 

# <codecell>

#------------#
# Parameters #
#------------#

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

# <codecell>

#-----------------#
# Human responses #
#-----------------#

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

##########################
##### MODEL OBSERVER #####
##########################

# model observer parameters
nthresh0          = 1                            # feedback threshold for "fall"
nthresh           = 1                            # ipe threshold for "fall"
nsamps            = 300                          # number of ipe samples
f_smooth          = True                         # smooth ipe likelihoods
p_ignore_stimulus = 0.0                          # probability of ignoring stimulus

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

# plt.figure()
# plt.imshow(exp**normalize(m_kappas[:, -1].T, axis=0)[1], cmap=cmap, interpolation='nearest', vmin=0, vmax=1, origin='lower')
# plt.xlabel("Feedback condition")
# plt.ylabel("Mass ratio")
# plt.title("Model observer belief as a function of feedback")
# cb = plt.colorbar(ticks=cticks)
# cb.set_ticklabels(logcticks)

plt.figure()
plt.plot(kappas, np.mean(m_responses, axis=1))
plt.xticks(plt.xticks()[0], np.round(10**plt.xticks()[0], decimals=2))
plt.title("Probability of model responding \"fall\"")
plt.xlabel("Feedback condition")
plt.ylim(0, 1)

# <codecell>

##################################
##### EXPERIMENTER INFERENCE #####
##################################

# particle filter parameters
n_part       = 500                               # number of particles
neffthresh   = n_part * 0.75                     # number of effective particles threshold
kw           = 0.1                               # kernel width for smoothing
pn           = 2.0                               # noise
f_use_model  = True                              # if True, use responses from model observer,
                                                 # if False, use responses from human data
f_plot_particle = False

# for smoothing likelihoods
smoother = mo.make_kde_smoother(np.arange(0, n_kappas*0.1, 0.1), kw)
# randomness
rso = np.random.RandomState(50)

# <codecell>

# "true" responses (for the purposes of inference)

if f_use_model:
    true_responses = m_responses.copy().astype('i8')
    # true_responses = mo.responses(
    #     m_kappas[:, [-1]]*np.ones(m_kappas.shape),
    #     outcomes[:, None], ipe_samps, p_ignore_stimulus, f_smooth)
    sidxs = [0, 3, 6, 8, 10, 13, 16, 18, 20, 23, 26]
    #sidxs = [0]#, 13, 26]

else:
    true_responses = h_responses.copy()
    sidxs = np.arange(true_responses.shape[0])

n_subj = len(sidxs)
print "Running inference on %d subjects" % n_subj

# <codecell>

# arrays to hold particles and weights
thetas = np.empty((n_subj, n_part, n_trial+1, n_kappas)) * np.nan
weights = np.empty((n_subj, n_part, n_trial+1)) * np.nan
mle_alphas = np.empty((n_subj, n_trial+1, n_kappas)) * np.nan

thetas[:, :, 0] = np.log(smoother(
    rso.dirichlet(np.ones(n_kappas), (n_subj, n_part))[..., None, :]))
#thetas[:, :, 0] = np.log(1. / n_kappas)
weights[:, :, 0] = np.log(np.ones((n_subj, n_part)) / n_part)
mle_alphas[:, 0] = np.ones((n_subj, n_kappas)) / n_kappas

# <codecell>

###########################
##### PARTICLE FILTER #####
###########################

if f_plot_particle:
    x = np.arange(n_kappas)
    plt.close('all')
    plt.figure(1)
    n = 4
    pp = []
    pl = []
    for i in xrange(n):
        plt.subplot(1, n, i+1)
        pp.append(plt.plot(x, np.zeros(n_kappas), 'b.')[0])
        pl.append(plt.plot(x, np.zeros(n_kappas), 'b-')[0])
        plt.ylim(0, 1)
    plt.show()
    plt.draw()
    plt.draw()

for idx, sidx in enumerate(sidxs):

    print idx
    rso.seed(100)
        
    for t in xrange(0, n_trial):

        thetas_t = thetas[idx, :, t].copy()
        weights_t = weights[idx, :, t].copy()
        samps_t = ipe_samps[t]

        # compute likelihood of outcomes
        p_outcomes = np.exp(mo.predict(
            thetas_t, outcomes[:, None], samps_t, f_smooth))

        # observe response
        response_t = int(true_responses[sidx, t])

        # compute likelihood of response
        p_response = (p_outcomes[:, response_t]*(1-p_ignore_stimulus) + 
                      (1./n_responses)*(p_ignore_stimulus))

        # observe feedback
        #obs_t = feedback[t, sidx]
        #obs_t = feedback[t, ikappa0]
        obs_t = feedback[t, -1]

        # update mass ratio belief
        ef = mo.evaluateFeedback(obs_t, samps_t, f_smooth)
        us_noise = pn*rso.dirichlet(np.ones(n_kappas), (n_part,))
        noise = smoother(us_noise[:, None, :])
        v = np.exp(normalize(thetas_t + ef + np.log(noise), axis=-1)[1])
        #m_theta = normalize(np.log(smoother(v[:, None])), axis=-1)[1]
        #m_theta = np.log(smoother(v[:, None]))
        m_theta = np.log(v)

        # weight particles
        weights_t = normalize(np.log(p_response) + weights_t)[1]
        
        # compute number of effective particles
        neff = 1. / np.sum(np.exp(weights_t)**2)

        # resample particles
        if neff < neffthresh:
            tidx = weightedSample(
                np.exp(weights_t), n_part, rso=rso)
            thetas[idx, :, t+1] = m_theta[tidx]
            weights[idx, :, t+1] = np.log(
                np.ones(n_part, dtype='f8') / n_part)
        else:
            thetas[idx, :, t+1] = m_theta
            weights[idx, :, t+1] = weights_t

        # plot
        if f_plot_particle:
            pp[0].set_ydata(np.exp(thetas_t)[0])
            pl[0].set_ydata(np.exp(thetas_t)[0])
            pp[1].set_ydata(np.exp(ef)[0])
            pl[1].set_ydata(np.exp(ef)[0])
            pp[2].set_ydata(us_noise[0])
            pl[2].set_ydata(noise[0])
            pp[3].set_ydata(v[0])
            pl[3].set_ydata(v[0])
            plt.draw()


# <codecell>

plt.close('all')
fig = plt.figure(3)
plt.clf()
#fig.set_figwidth(9)
#fig.set_figheight(3)
#r, c = 1, 3
r, c = 3, 4
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
        np.sum(np.exp(thetas[i]) * np.exp(weights[i,...,None]), axis=0),
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
plt.draw()
plt.draw()

    # subjname = "Model Subj. r=%.1f" % ratios[sidx]
    # print subjname
    # mle_theta = np.mean(np.exp(thetas[sidx]), axis=0)
    # plt.figure(2)
    # lat.plot_theta(
    #     3, 4, sidx+1,
    #     mle_theta, subjname, exp=np.e, ratios=ratios)

# <codecell>

print thetas.shape
print weights.shape

