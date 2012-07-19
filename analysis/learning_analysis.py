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

    obs0 = mo.ModelObserver(
        ime_samples=model0[0, 1].copy(),
        kappas=kappas, n_F=11)
    obs1 = mo.ModelObserver(
        ime_samples=model1[0, 1].copy(),
        kappas=kappas, n_F=11)

    ime0 = obs0.IME(slice(None)).copy()
    ime1 = obs1.IME(slice(None)).copy()

    # make sure the IME calculations are the same
    assert (np.abs(np.mean(ime0, axis=0) - np.mean(ime1, axis=0)) < 1e-6).all()

    S0 = stimuli0[0].copy()
    F0 = model0[0, 0, :, kidx, 0].copy()
    S1 = stimuli1[0].copy()
    F1 = model1[0, 0, :, kidx, 0].copy()

    pkappa0 = obs0.learningCurve(F0)[1][-1]
    pkappa1 = obs1.learningCurve(F1)[1][-1]
    kdiff = np.abs(pkappa0 - pkappa1)

    # make sure beliefs about kappa are independent of order
    assert (kdiff < 1e-6).all()

######################################################################

pltkwargs = {
    'color': 'k',
    'align': 'center'
    }

human, stimuli, sort, model = order_by_trial(
    rawhuman, rawhstim, raworder, fellsamp)

n_T = stimuli.shape[1]
n_F = 11
n_R = 7
n_kappa = len(kappas)
n_total_samp = model.shape[4]
n_samp = 100

RSO = np.random.RandomState(0)
sidx = RSO.randint(0, n_total_samp, (n_T, n_kappa, n_samp))
ime_samps = npl.choose(sidx, model[0, 1, :, :, :, None], axis=2)

S   = stimuli[0].copy()
#F   = (model[0, 0, :, :, 0] > 0).astype('int')
#ime = (ime_samps > 0).astype('int')
F   = model[0, 0, :, :, 0].copy()
ime = ime_samps.astype('int')

plt.close('all')
fig1 = plt.figure()
plt.suptitle("Posterior P(kappa|F)")
fig2 = plt.figure()
plt.suptitle("Likelihood P(F|kappa)")

for kidx, ratio in enumerate(ratios):
    
    obs = mo.ModelObserver(
        ime_samples=ime,
        kappas=kappas,
        N=1,
        Cf=10,
        Cr=6,
        n_F=11,
        smooth=False)
    lh, joint, Pt_kappa = obs.learningCurve(F[:, kidx])

    ax = fig1.add_subplot(4, 3, kidx+1)
    ax.cla()
    ax.imshow(
        np.exp(Pt_kappa.T),
        aspect='auto', interpolation='nearest',
        vmin=0, vmax=1, cmap='hot')
    ax.set_xticks([])
    ax.set_ylabel("Mass Ratio")
    ax.set_yticks(np.arange(n_kappa))
    ax.set_yticklabels(ratios)
    ax.set_title("ratio = %.1f" % ratio)

    ax = fig2.add_subplot(4, 3, kidx+1)
    ax.cla()
    ax.imshow(
        np.exp(lh.T),
        aspect='auto', interpolation='nearest',
        vmin=0, vmax=1, cmap='gray')
    ax.set_xticks([])
    ax.set_ylabel("Mass Ratio")
    ax.set_yticks(np.arange(n_kappa))
    ax.set_yticklabels(ratios)
    ax.set_title("ratio = %.1f" % ratio)

plt.draw()

# obs = mo.ModelObserver(
#     ime_samples=model[0, 1].copy(),
#     kappas=kappas,
#     N=1,
#     Cf=10,
#     Cr=6)

# P_theta = rvs.Dirichlet(np.ones(n_kappa))
# R = 6 - human.copy()
# thetas = P_theta.sample((1000, n_kappa))

# # P(R_t | S_t, theta_t)
# for t in xrange(n_T):

#     mn = rvs.Multinomial(5, thetas)
#     steps = mn.sample()

#     P_theta_new = P_theta.updateMultinomial(steps)

#     Rt = R[0, t]
#     sRt = np.empty(thetas.shape[0])
#     for i in xrange(thetas.shape[0]):
#         rtheta = np.clip(thetas[i] + np.random.normal(0, 0.05, n_kappa), 0, 1)
#         theta = rtheta / np.sum(rtheta)
#         obs.thetas[t] = theta
#         sRt[i] = obs.generateResponse()
#     P_Rt = sRt == Rt
#     print np.mean(P_Rt)

# # P(theta_t | R_t, theta_t-1)




# N = np.arange(-3, 4, 1)
# Cf = np.arange(2, 31, 2)
# Cr = np.arange(1, 31, 2)
# params = npl.vgrid(N, Cf, Cr).T
# errs = []

# def compute_response(lh_F, N, Cf, Cr):
#     ## Calculate loss
#     loss = L(outcomes[:, None], responses[None, :], N=N, Cf=Cf, Cr=Cr)
#     risk = np.sum(loss[:, :, None, None] * np.exp(lh_F[:, None]), axis=0)
#     ## Compute optimal responses
#     mR = np.argmin(risk, axis=0)
#     return mR

# bcR = np.sum(R[None, :, :] == responses[:, None, None], axis=-1)
# print len(params)
# for pidx in xrange(len(params)):
#     if pidx % 100 == 0:
#         print pidx
#     n, cf, cr = params[pidx]
#     mR = compute_response(lh_F, n, cf, cr)
#     bcmR = np.sum(mR[None, :, :] == responses[:, None, None], axis=-1)
#     se = np.sum((bcmR - bcR) ** 2, axis=0)
#     errs.append(se)

# best = params[np.argmin(errs, axis=0)]

# for sidx in xrange(n_subj):
#     mR = compute_response(lh_F[:, [sidx]], *best[sidx])
#     print "model ", np.bincount(mR[0].astype('int'))
#     print "people", np.bincount(R[sidx].astype('int'))
#     print
