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

mthresh = 0.095
zscore = False
ALPHAS = np.logspace(np.log10(0.0001), np.log10(120), 200)
SEED = 0
N_BOOT = 10000

def order_by_trial(human, stimuli, order, model):

    n_subjs = human.shape[1]
    n_stim = stimuli.size
    n_rep = human.shape[2]

    htrial = human.transpose((1, 0, 2)).reshape((n_subjs, -1))
    horder = order.transpose((1, 0, 2)).reshape((n_subjs, -1))

    trial_human = []
    trial_order = []
    trial_stim = []
    trial_model = []

    shape = np.ones((n_stim, n_rep))
    sidx = list((np.arange(n_stim)[:, None] * shape).ravel())
    
    for hidx in xrange(n_subjs):
        hd = htrial[hidx].copy()
        ho = horder[hidx].copy()
        
        sort = np.argsort(ho)
        shuman = hd[sort]
        sorder = ho[sort]
        sstim = stimuli[..., sidx, :, :][..., sort, :, :]
        smodel = model[..., sidx, :, :][..., sort, :, :]
        
        trial_human.append(shuman)
        trial_order.append(sort)
        trial_stim.append(sstim)
        trial_model.append(smodel)
        
    trial_human = np.array(trial_human, copy=True)
    trial_order = np.array(trial_order, copy=True)
    trial_stim = np.array(trial_stim, copy=True)
    trial_model = np.array(trial_model, copy=True)

    out = (trial_human, trial_stim, trial_order, trial_model)
    return out


# Human
rawhuman, rawhstim, rawhmeta, raworder = tat.load_human(
    exp_ver=7, return_order=True)
n_subjs, n_reps = rawhuman.shape[1:]

# Model
if os.path.exists("truth_samples.npy"):
    truth_samples = np.load("truth_samples.npy")
else:
    rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=4)
    sigmas = rawsmeta["sigmas"]
    phis = rawsmeta["phis"]
    kappas = rawsmeta["kappas"]
    sigma0 = list(sigmas).index(0)
    phi0 = list(phis).index(0)
    rawmodel0 = rawmodel[sigma0, phi0][None, None]
    pfell, nfell, truth_samples = tat.process_model_stability(
        rawmodel0, mthresh=mthresh, zscore=zscore, pairs=False)
    np.save("truth_samples.npy", truth_samples)

rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=6)
sigmas = rawsmeta["sigmas"]
phis = rawsmeta["phis"]
kappas = rawsmeta["kappas"]
ratios = np.round(10 ** kappas, decimals=1)
pfell, nfell, fell_sample = tat.process_model_stability(
    rawmodel, mthresh=mthresh, zscore=zscore, pairs=False)

all_model = np.array([truth_samples, fell_sample])
fellsamp = all_model[:, 0, 0].transpose((0, 2, 1, 3)).astype('int')

assert (rawhstim == rawsstim).all()

def random_order(n, shape1, shape2, axis=-1, seed=0):
    tidx = np.arange(n)
    RSO = np.random.RandomState(seed)
    RSO.shuffle(tidx)
    stidx = tidx.reshape(shape1)
    order = stidx * np.ones(shape2)
    return order

######################################################################
## Error checking

nstim, nsubj, nrep = raworder.shape
ntrial = nstim * nrep

raworder0 = random_order(ntrial, (nstim, 1, nrep), raworder.shape)
raworder1 = random_order(ntrial, (nstim, 1, nrep), raworder.shape)
human0, stimuli0, sort0, model0 = order_by_trial(
    rawhuman, rawhstim, raworder0, fellsamp)
human1, stimuli1, sort1, model1 = order_by_trial(
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
        n_outcomes=11)
    lh1, joint1, pkappa1 = mo.ModelObserver(
        ime_samples=model1[0, 1].copy(),
        truth=F1.copy(),
        n_outcomes=11)

    kdiff = np.abs(pkappa0 - pkappa1)

    # make sure beliefs about kappa are independent of order
    assert (kdiff < 1e-6).all()

######################################################################

# human, stimuli, sort, model = order_by_trial(
#     rawhuman, rawhstim, raworder, fellsamp)

# n_trial = stimuli.shape[1]
# n_kappas = len(kappas)
# n_total_samp = model.shape[4]
# n_samp = 48

# #RSO = np.random.RandomState(0)
# #sidx = RSO.randint(0, n_total_samp, (n_trial, n_kappas, n_samp))
# #ime_samps = npl.choose(sidx, model[0, 1, :, :, :, None], axis=2)

# ime_samps = model[0, 1].copy().astype('int')
# truth = model[0, 0, :, :, 0].copy().astype('int')

# exp = 1.2

# plt.close('all')
# fig1 = plt.figure()
# plt.suptitle("Posterior P(kappa|F)")
# fig2 = plt.figure()
# plt.suptitle("Likelihood P(F|kappa)")

# for kidx, ratio in enumerate(ratios):
    
#     lh, joint, Pt_kappa = mo.ModelObserver(
#         ime_samples=ime_samps,
#         truth=truth[:, kidx],
#         n_outcomes=11)
#     nlh = np.exp(lh.T) / np.sum(np.exp(lh.T), axis=0)

#     ax = fig1.add_subplot(4, 3, kidx+1)
#     ax.cla()
#     ax.imshow(
#         #np.exp(Pt_kappa.T),
#         exp ** Pt_kappa.T,
#         aspect='auto', interpolation='nearest',
#         vmin=0, vmax=1, cmap='hot')
#     ax.set_xticks([])
#     ax.set_ylabel("Mass Ratio")
#     ax.set_yticks(np.arange(n_kappas))
#     ax.set_yticklabels(ratios)
#     ax.set_title("ratio = %.1f" % ratio)

#     ax = fig2.add_subplot(4, 3, kidx+1)
#     ax.cla()
#     ax.imshow(
#         nlh,
#         aspect='auto', interpolation='nearest',
#         vmin=0, vmax=nlh.max(), cmap='gray')
#     ax.set_xticks([])
#     ax.set_ylabel("Mass Ratio")
#     ax.set_yticks(np.arange(n_kappas))
#     ax.set_yticklabels(ratios)
#     ax.set_title("ratio = %.1f" % ratio)

# plt.draw() 

######################################################################

def plot_theta(theta, idx, title):
    plt.subplot(4, 4, idx)
    plt.cla()
    plt.imshow(
        exp ** np.log(theta.T),
        aspect='auto', interpolation='nearest',
        vmin=0, vmax=1, cmap='hot')
    plt.xticks([], [])
    plt.ylabel("Mass Ratio")
    plt.yticks(np.arange(n_kappas), ratios)
    plt.title(title)
    plt.draw()

exp = 1.2
ratio = 10
kidx = list(ratios).index(ratio)

human, stimuli, sort, model = order_by_trial(
    rawhuman, rawhstim, raworder, fellsamp)

n_trial = stimuli.shape[1]
n_kappas = len(kappas)
n_part = 50
n_responses = 7

# compute probability of each outcome given kappa for each type of
# predicate (either the number that fell or whether it fell)
ime_samps = model[0, 1].copy().astype('int')
p_outcomes_nfell = mo.IME(ime_samps, 11)
p_outcomes_pfell = mo.IME((ime_samps > 0).astype('int'), 2)
p_outcomes = np.empty((2,) + p_outcomes_nfell.shape)
p_outcomes.fill(-np.inf)
p_outcomes[0] = p_outcomes_nfell
p_outcomes[1, :, :, :2] = p_outcomes_pfell

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
mo_lh, mo_joint, mo_theta = mo.ModelObserver(
    ime_samples=ime_samps,
    truth=truth_nfell,
    n_outcomes=11)
mo_responses = mo.response(
    mo_theta[:-1], p_outcomes_nfell, loss_nfell)[None]
# actual human responses on a scale from 0 to 6
subj_responses = 6 - human.copy()
responses = np.concatenate([mo_responses, subj_responses], axis=0)
n_subj = responses.shape[0]

# arrays to hold particles and weights
thetas = np.empty((n_subj, n_part, n_trial+1, n_kappas)) * np.nan
weights = np.empty((n_subj, n_part, n_trial+1)) * np.nan

# initial particle values and weights -- sample values from a uniform
# Dirichlet prior
P_theta0 = rvs.Dirichlet(np.ones((n_part, n_kappas)))
thetas[:, :, 0] = np.log(P_theta0.sample((n_subj, n_part, n_kappas)))
weights[:, :, 0] = np.log(np.ones(n_part) / float(n_part))

# predicate indicator
P_predicate = rvs.Bernoulli(p = 0.5)

plot_theta(np.exp(mo_theta), 1, "Model Observer (true)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)

for sidx in xrange(n_subj):
    if sidx > 0:
        subjname = "Subject %d" % sidx
    else:
        subjname = "Model Observer"
    print subjname

    rso = np.random.RandomState(100)
    P_predicate.reseed()
        
    for t in xrange(0, n_trial):

        # sample new particles
        tidx = stats.weightedSample(
            np.exp(weights[sidx, :, t]), n_part, rso=rso)
        new_thetas = thetas[sidx, :, t][tidx][:, None]

        # choose predicate
        pidx = P_predicate.sample(n_part).astype('int')
        n_outcomes = (pidx * 9) + 2

        loss_t = np.choose(pidx[:, None, None], loss)[:, None]
        p_outcomes_t = np.choose(
            pidx[:, None, None, None], p_outcomes[:, [t]])
        truth_t = np.choose(pidx[:, None], truth[:, [t]])

        # compute responses
        m_response = mo.response(new_thetas, p_outcomes_t, loss_t)
        m_lh, m_joint, m_theta = mo.learningCurve(
            truth_t, new_thetas, p_outcomes_t)

        mR = m_response[:, 0]
        mT = m_theta[:, 1]

        # calculate weights
        w = np.ones((n_part, n_responses))
        w[mR[:, None] == np.arange(n_responses)] += 10
        p = np.log(w / np.expand_dims(np.sum(w, axis=-1), axis=-1))
        weights_t = stats.normalize(p[:, responses[sidx, t]])[1]

        # update
        thetas[sidx, :, t+1] = mT
        weights[sidx, :, t+1] = weights_t

    mle_theta = np.mean(np.exp(thetas[sidx]), axis=0)
    mle_theta /= np.sum(mle_theta, axis=-1)[..., None]
    plot_theta(mle_theta, sidx+2, subjname)

    msubj_responses = mo.response(
        #np.log(mle_theta[[-1]]), p_outcomes_nfell, loss_nfell)
        np.log(mle_theta[1:]), p_outcomes_nfell, loss_nfell)
    subj_responses = responses[sidx]

    err0 = np.mean((subj_responses - mo_responses) ** 2)
    err1 = np.mean((subj_responses - msubj_responses) ** 2)
    print "%.2f --> %.2f (% .2f)" % (err0, err1, err1-err0)


plot_theta(np.exp(mo_theta), 1, "Model Observer (true)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)

for sidx in xrange(n_subj):
    if sidx > 0:
        subjname = "Subject %d" % sidx
    else:
        subjname = "Model Observer"
    
    mle_theta = np.mean(np.exp(thetas[sidx]), axis=0)
    mle_theta /= np.sum(mle_theta, axis=-1)[..., None]
    plot_theta(mle_theta, sidx+2, subjname)

    msubj_responses = mo.response(
        #np.log(mle_theta[[-1]]), p_outcomes_nfell, loss_nfell)
        np.log(mle_theta[1:]), p_outcomes_nfell, loss_nfell)
    subj_responses = responses[sidx]

    err0 = np.mean((subj_responses - mo_responses) ** 2)
    err1 = np.mean((subj_responses - msubj_responses) ** 2)
    print "%.2f --> %.2f (% .2f)" % (err0, err1, err1-err0)



