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

    lh0, joint0, pkappa0 = mo.ModelObserver(
        ime_samples=model0[0, 1].copy(),
        truth=F0.copy(),
        n_outcomes=11,
        exp=1e10)
    lh1, joint1, pkappa1 = mo.ModelObserver(
        ime_samples=model1[0, 1].copy(),
        truth=F1.copy(),
        n_outcomes=11,
        exp=1e10)

    kdiff = np.abs(pkappa0 - pkappa1)

    # make sure beliefs about kappa are independent of order
    assert (kdiff < 1e-6).all()

######################################################################
# Model observers for each true mass ratio

human, stimuli, sort, model = lat.order_by_trial(
    rawhuman, rawhstim, raworder, fellsamp)

reload(mo)
reload(lat)

# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)

# samples from the IME
ime_samps = model[0, 1].copy().astype('int')
# true outcomes for each mass ratio
truth = model[0, 0, :, :, 0].copy().astype('int')
# probability of each outcome given theta
p_outcomes = mo.IME(ime_samps, 11)
# loss function
loss = mo.Loss(11, 7, N=2, Cf=10, Cr=6)

# arrays to hold the model observer data
model_lh = np.empty((n_kappas, n_trial+1, n_kappas))
model_joint = np.empty((n_kappas, n_trial+1, n_kappas))
model_theta = np.empty((n_kappas, n_trial+1, n_kappas))
model_subjects = np.empty((n_kappas, n_trial)).astype('int')

for kidx, ratio in enumerate(ratios):
    # compute belief over time
    lh, joint, theta = mo.ModelObserver(
        ime_samples=ime_samps,
        truth=truth[:, kidx],
        n_outcomes=11,
        exp=1e10)
    # compute responses
    response = mo.response(theta[:-1], p_outcomes, loss)[None]
    # store data
    model_lh[kidx] = lh
    model_joint[kidx] = joint
    model_theta[kidx] = theta
    model_subjects[kidx] = response

# plot it
plt.figure(1)
plt.clf()
plt.suptitle("Posterior P(kappa|F)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
for kidx, ratio in enumerate(ratios):
    subjname = "Model Subj. r=%.1f" % ratios[kidx]
    lat.plot_theta(
        np.exp(model_theta[kidx]),
        kidx+1, subjname, exp=np.e, ratios=ratios)

######################################################################

ratio = 10
kidx  = list(ratios).index(ratio)
exp   = np.e

n_trial     = stimuli.shape[1]
n_kappas    = len(kappas)
n_part      = 100
n_responses = 7

# compute probability of each outcome given kappa for each type of
# predicate (either the number that fell or whether it fell)
ime_samps = model[0, 1].copy().astype('int')
p_outcomes_nfell = mo.evaluateObserved(
    mo.IME(ime_samps, 11), exp=exp)
p_outcomes_pfell = mo.evaluateObserved(
    mo.IME((ime_samps > 0).astype('int'), 2), exp=exp)
p_outcomes = np.empty((2,) + p_outcomes_nfell.shape)
p_outcomes.fill(-np.inf)
p_outcomes[0] = p_outcomes_nfell
p_outcomes[1, :, :, :2] = p_outcomes_pfell

# observation density
p_obs = mo.observationDensity(11, exp=exp)

# truth values for each type of predicate
truth_nfell = model[0, 0, :, kidx, 0].copy().astype('int')
truth_pfell = (truth_nfell > 0).astype('int')
truth = np.array([truth_nfell, truth_pfell])

# loss for each type of predicate
loss_nfell = mo.Loss(11, 7, N=2, Cf=10, Cr=6)
loss_pfell = mo.Loss(2, 7, N=0, Cf=1, Cr=1)
loss = np.empty((2,) + loss_nfell.shape)
loss.fill(np.inf)
loss[0] = loss_nfell
loss[1, :2] = loss_pfell

# model observer responses
mo_lh = model_lh[kidx].copy()
mo_joint = model_joint[kidx].copy()
mo_theta = model_theta[kidx].copy()
mo_responses = model_subjects[kidx][None].copy()
# actual human responses on a scale from 0 to 6
subj_responses = 6 - human.copy()
# responses = np.concatenate([mo_responses, subj_responses], axis=0)
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

        # choose predicate
        pidx = P_predicate.sample(n_part).astype('int')

        #loss_t = np.choose(pidx[:, None, None], loss)[:, None]
        loss_t = np.choose(pidx[:, None, None]*0, loss_nfell[None])[:, None]
        p_nfell_t = np.choose(
            pidx[:, None, None, None]*0, p_outcomes_nfell[None][:, [t]])
        p_outcomes_t = np.choose(
            pidx[:, None, None, None], p_outcomes[:, [t]])
        truth_t = np.choose(pidx[:, None], truth[:, [t]])
        p_obs_t = np.choose(
            truth_t[:, :, None], p_obs.T[:, None, None, :])
        obs_t = stats.weightedSample(np.exp(p_obs_t), 1, axis=-1)[..., 0]

        # compute responses
        #m_response = mo.response(thetas_t, p_outcomes_t, loss_t)
        m_response = mo.response(thetas_t, p_nfell_t, loss_t)
        m_lh, m_joint, m_theta = mo.learningCurve(
            #truth_t, thetas_t, p_outcomes_t)
            obs_t, thetas_t, p_outcomes_t)

        mR = m_response[:, 0].copy()
        mT = m_theta[:, 1].copy()

        # calculate weights
        w = np.ones((n_part, n_responses))
        #w = np.zeros((n_part, n_responses))
        #w[mR[:, None] == np.arange(n_responses)] += 1e15
        w[mR[:, None] == np.arange(n_responses)] += 1
        #w = np.exp2(-np.abs(mR[:, None] - np.arange(n_responses)))
        p = np.log(w / np.expand_dims(np.sum(w, axis=-1), axis=-1))
        weights_t = stats.normalize(p[:, true_responses[sidx, t]])[1]

        # sample new particles
        tidx = stats.weightedSample(
            np.exp(weights_t), n_part, rso=rso)

        #pdb.set_trace()

        # update
        thetas[sidx, :, t+1] = mT[tidx]
        weights[sidx, :, t+1] = np.log(np.ones(n_part, dtype='f8') / n_part)

        if t % 25 == 0:
            mle_theta = np.mean(np.exp(thetas[sidx, :, t+1]), axis=0)
            print sidx, t, np.round(mle_theta, decimals=2)


plt.figure(2)
plt.clf()
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
for sidx in xrange(n_subj):
    subjname = "Model Subj. r=%.1f" % ratios[sidx]
    print subjname
    mle_theta = np.mean(np.exp(thetas[sidx]), axis=0)
    plt.figure(2)
    lat.plot_theta(mle_theta, sidx+1, subjname, exp=2, ratios=ratios)
    msubj_responses = mo.response(
        np.log(mle_theta[1:]), p_outcomes_nfell, loss_nfell)
    subj_responses = true_responses[sidx]
    err0 = np.mean((subj_responses - mo_responses) ** 2)
    err1 = np.mean((subj_responses - msubj_responses) ** 2)
    print "%.2f --> %.2f (% .2f)" % (err0, err1, err1-err0)
