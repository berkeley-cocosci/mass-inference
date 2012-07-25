import collections
import matplotlib.cm as cm
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
import cogphysics.lib.stats as stats
import cogphysics.tower.analysis_tools as tat

from matplotlib import rc
from sklearn import linear_model

from cogphysics.lib.corr import xcorr

import model_observer as mo
import dirichlet
import learning_analysis_tools as lat


rawhuman, rawhstim, raworder, fellsamp, params = lat.load()
sigmas, phis, kappas = params
ratios = np.round(10 ** kappas, decimals=1)

######################################################################
## Error checking

nstim, nsubj, nrep = raworder.shape
ntrial = nstim * nrep

raworder0 = lat.random_order(ntrial, (nstim, 1, nrep), raworder.shape)
raworder1 = lat.random_order(ntrial, (nstim, 1, nrep), raworder.shape)
human0, stimuli0, sort0, model0 = lat.order_by_trial(
    rawhuman, rawhstim, raworder0, fellsamp)
human1, stimuli1, sort1, model1 = lat.order_by_trial(
    rawhuman, rawhstim, raworder1, fellsamp)

# make sure trials got reordered correctly
assert (np.mean(human0, axis=-1) == np.mean(human1, axis=-1)).all()
assert (np.mean(model0, axis=2) == np.mean(model1, axis=2)).all()

for kidx, ratio in enumerate(ratios):

    F0 = model0[0, 0, :, kidx, 0]
    F1 = model1[0, 0, :, kidx, 0]
    br0 = np.mean(F0.ravel()[:, None] == np.arange(11), axis=0)
    br1 = np.mean(F0.ravel()[:, None] == np.arange(11), axis=0)

    lh0, joint0, pkappa0 = mo.ModelObserver(
        ime_samples=model0[0, 1].copy(),
        truth=F0.copy(),
        baserate=br0,
        n_outcomes=11,
        A=1e10)
    lh1, joint1, pkappa1 = mo.ModelObserver(
        ime_samples=model1[0, 1].copy(),
        truth=F1.copy(),
        baserate=br1,
        n_outcomes=11,
        A=1e10)

    kdiff = np.abs(pkappa0 - pkappa1)

    # make sure beliefs about kappa are independent of order
    assert (kdiff < 1e-6).all()

######################################################################

human, stimuli, sort, model = lat.order_by_trial(
    rawhuman, rawhstim, raworder, fellsamp)

reload(mo)
reload(lat)

# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)
n_outcomes   = 11

# samples from the IME
ime_samps = model[0, 1].copy().astype('int')
#ime_samps = model[0, 0][:, :, [0]].copy().astype('int')
# true outcomes for each mass ratio
truth = model[0, 0, :, :, 0].copy().astype('int')
# loss function
#loss = mo.Loss(11, 7, N=2, Cf=10, Cr=6)
loss = mo.Loss(11, 7, N=0, Cf=5, Cr=6)
# base outcome rate
counts = np.sum(truth[..., None] == np.arange(11), axis=0) + 1
baserate = stats.normalize(np.log(counts), axis=-1)[1]


# # differences in feedback
# plt.figure(100)
# plt.clf()
# msd = np.mean(np.abs(truth[:, :, None] - truth[:, None, :]), axis=0)
# plt.imshow(msd, cmap='gray', interpolation='nearest', vmin=0, vmax=5)
# plt.title("Differences in feedback for different kappas")
# plt.xticks(np.arange(n_kappas), ratios)
# plt.yticks(np.arange(n_kappas), ratios)

# # differences in model predictions
# plt.figure(101)
# plt.clf()
# ime_mean = np.mean(ime_samps, axis=-1)
# msd = np.mean(np.abs(ime_mean[:, :, None] - ime_mean[:, None, :]), axis=0)
# plt.imshow(msd, cmap='gray', interpolation='nearest', vmin=0, vmax=5)
# plt.title("Differences in model predictions for different kappas")
# plt.xticks(np.arange(n_kappas), ratios)
# plt.yticks(np.arange(n_kappas), ratios)

# # mean number of blocks that fall for each kappa
# plt.figure(102)
# plt.clf()
# meanblocks0 = np.mean(truth, axis=0)
# meanblocks = np.mean(
#     ime_samps.transpose((1, 0, 2)).reshape((n_kappas, -1)), axis=-1)
# plt.plot(ratios, meanblocks0, label="truth")
# plt.plot(ratios, meanblocks, label="model")
# plt.xticks(ratios, ratios)
# plt.title("Mean number of blocks that fall for all stimuli")
# plt.legend()

# # base rates for the model
# plt.figure(103)
# plt.clf()
# mbr = np.sum(
#     ime_samps.transpose((1, 0, 2)).reshape(
#         (n_kappas, -1))[..., None] == np.arange(11),
#     axis=1)
# mbr = mbr / np.sum(mbr, axis=-1).astype('f8')[..., None]
# plt.imshow(mbr, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
# plt.title("Rates of outcomes for the model")
# plt.xticks(np.arange(11), np.arange(11))
# plt.yticks(np.arange(n_kappas), ratios)

######################################################################
# Model observers for each true mass ratio

# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)

# arrays to hold the model observer data
model_lh = np.empty((n_kappas, n_trial+1, n_kappas))
model_joint = np.empty((n_kappas, n_trial+1, n_kappas))
model_theta = np.empty((n_kappas, n_trial+1, n_kappas))
model_subjects = np.empty((n_kappas, n_trial)).astype('int')

for kidx, ratio in enumerate(ratios):
    # compute belief over time
    lh, joint, theta, response = mo.ModelObserver(
        ime_samples=ime_samps,
        truth=truth[:, kidx],
        n_outcomes=11,
        baserate=baserate[kidx],
        A=1e10,
        loss=loss)
    # store data
    model_lh[kidx] = lh.copy()
    model_joint[kidx] = joint.copy()
    model_theta[kidx] = theta.copy()
    model_subjects[kidx] = response.copy()

# plot it
plt.figure(1)
plt.clf()
plt.suptitle("Posterior P(kappa|F)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
for kidx, ratio in enumerate(ratios):
    subjname = "Model Subj. r=%.1f" % ratios[kidx]
    lat.plot_theta(
        3, 4, kidx+1,
        np.exp(model_theta[kidx]),
        subjname, exp=np.e, ratios=ratios)

plt.figure(2)
plt.clf()
plt.suptitle("Likelihood P(F|kappa)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
for kidx, ratio in enumerate(ratios):
    subjname = "Model Subj. r=%.1f" % ratios[kidx]
    lat.plot_theta(
        3, 4, kidx+1,
        np.exp(model_lh[kidx]) / np.sum(np.exp(model_lh[kidx]), axis=-1)[..., None],
        subjname,
        exp=1.3,
        ratios=ratios,
        cmap='gray')

######################################################################

ratio = 10
kidx  = list(ratios).index(ratio)
A     = 3#np.e  # observation uncertainty
B     = 10   # mixing parameter

n_trial     = stimuli.shape[1]
n_kappas    = len(kappas)
n_part      = 100
n_responses = 7

# samples from the IME
p_outcomes = mo.IME(ime_samps, 11, baserate[kidx])
p_obs = mo.observationDensity(11, A)[0]

# model observer responses
mo_lh = model_lh[kidx].copy()
mo_joint = model_joint[kidx].copy()
mo_theta = model_theta[kidx].copy()
mo_responses = model_subjects[kidx][None].copy()

# actual human responses on a scale from 0 to 6
#true_responses = 6 - human.copy()
true_responses = model_subjects.copy()
n_subj = true_responses.shape[0]

# arrays to hold particles and weights
thetas = np.empty((n_subj, n_part, n_trial+1, n_kappas)) * np.nan
weights = np.empty((n_subj, n_part, n_trial+1)) * np.nan
mle_alphas = np.empty((n_subj, n_trial+1, n_kappas)) * np.nan

# initial particle values and weights -- sample values from a uniform
# Dirichlet prior
P_theta0 = rvs.Dirichlet(np.ones((n_part, n_kappas)))
thetas[:, :, 0] = np.log(P_theta0.sample((n_subj, n_part, n_kappas)))
weights[:, :, 0] = np.log(np.ones((n_subj, n_part)) / n_part)
mle_alphas[:, 0] = np.ones((n_subj, n_kappas)) / n_kappas

# predicate indicator
P_predicate = rvs.Bernoulli(p = 0.0)

for sidx in xrange(n_subj):

    rso = np.random.RandomState(100)
    P_predicate.reseed()
        
    for t in xrange(0, n_trial):

        thetas_t = thetas[sidx, :, t].copy()[:, None]
        weights_t = weights[sidx, :, t].copy()

        truth_t = truth[None, t, kidx]
        p_outcomes_t = p_outcomes[None, t]
        response_t = true_responses[sidx, t]

        # sample observations
        p_obs_t = p_obs[:, truth_t]
        obs_t = stats.weightedSample(
            np.exp(p_obs_t), n_part, axis=0, rso=rso).T

        # compute responses
        m_response = mo.response(thetas_t, p_outcomes_t, loss)
        m_lh, m_joint, m_theta = mo.learningCurve(
            obs_t, thetas_t, p_outcomes_t)

        # calculate weights
        w = np.ones((n_part, n_responses))
        w[m_response[:, [0]] == np.arange(n_responses)] += B
        p = np.log(w / np.expand_dims(np.sum(w, axis=-1), axis=-1))
        weights_t = stats.normalize(p[:, response_t])[1]

        # sample new particles
        tidx = stats.weightedSample(
            np.exp(weights_t), n_part, rso=rso)

        # update
        thetas[sidx, :, t+1] = m_theta[:, 1][tidx]
        weights[sidx, :, t+1] = np.log(np.ones(n_part, dtype='f8') / n_part)

        if (t % 25 == 0) or (t == n_trial-1):
            mle_theta = np.mean(np.exp(thetas[sidx, :, t+1]), axis=0)
            print sidx, t, np.round(mle_theta, decimals=2)


plt.figure(3)
plt.clf()
plt.suptitle("MLE P(kappa)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
for sidx in xrange(n_subj):
    subjname = "Model Subj. r=%.1f" % ratios[sidx]
    print subjname
    mle_theta = np.mean(np.exp(thetas[sidx]), axis=0)
    plt.figure(2)
    lat.plot_theta(
        3, 4, sidx+1,
        mle_theta, subjname, exp=np.e, ratios=ratios)
    msubj_responses = mo.response(
        np.log(mle_theta[1:]), p_outcomes, loss)
    subj_responses = true_responses[sidx]
    err0 = np.mean((subj_responses - mo_responses) ** 2)
    err1 = np.mean((subj_responses - msubj_responses) ** 2)
    print "%.2f --> %.2f (% .2f)" % (err0, err1, err1-err0)
