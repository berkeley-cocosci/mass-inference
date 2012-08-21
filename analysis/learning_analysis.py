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
import cogphysics.tower.analysis_tools as tat

from matplotlib import rc
from sklearn import linear_model

from cogphysics.lib.corr import xcorr

import model_observer as mo
import learning_analysis_tools as lat

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

######################################################################
## Load and process data

reload(mo)
reload(lat)

rawhuman, rawhstim, raworder, msamp, params = lat.load('stability')
sigmas, phis, kappas = params
ratios = np.round(10 ** kappas, decimals=1)

human, stimuli, sort, model = lat.order_by_trial(
    rawhuman, rawhstim, raworder, msamp)

predicates = list(model.dtype.names)
predicates.remove('stability_pfell')
#predicates.remove('direction')
predicates.remove('stability_nfell')
predicates.remove('radius')

# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)
#n_outcomes   = (11, 8)
#n_outcomes   = (5, 8)
#n_outcomes   = (11,)
n_outcomes   = (16,)
n_predicate  = len(predicates)

# samples from the IME
ime_samps = model[0, 1].copy()
#ime_samps = model[0, 0][:, :, [0]].copy()
# true outcomes for each mass ratio
truth = model[0, 0, :, :, 0].copy()

# loss function
loss = mo.Loss(n_outcomes, 7, predicates, N=0, Cf=5, Cr=6)

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
        ime_samples=ime_samps.copy(),
        feedback=truth[:, kidx].copy(),
        n_outcomes=n_outcomes,
        predicates=predicates,
        loss=loss)

    # store data
    model_lh[kidx] = lh.copy()
    model_joint[kidx] = joint.copy()
    model_theta[kidx] = theta.copy()
    model_subjects[kidx] = response.copy()

# plot it
plt.figure(10)
plt.clf()
plt.suptitle("Posterior P(kappa|F)")
plt.subplots_adjust(wspace=0.3, hspace=0.2, left=0.1, right=0.9, top=0.9, bottom=0.1)
for kidx, ratio in enumerate(ratios):
    subjname = "Model Subj. r=%.1f" % ratios[kidx]
    lat.plot_theta(
        3, 4, kidx+1,
        np.exp(model_theta[kidx]),
        subjname,
        exp=1.3,
        ratios=ratios)

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
B     = 10   # mixing parameter

n_trial     = stimuli.shape[1]
n_kappas    = len(kappas)
n_part      = 100
n_responses = 7

# samples from the IME
p_outcomes = mo.IME(ime_samps, n_outcomes, predicates)

import kde
#from kde import gen_direction_edges
#edges, binsize, offset = gen_direction_edges(16)
fig, axes = plt.subplots(3, 4)#, subplot_kw=dict(polar=True))

# def hist(ax, e, z, s, t, title=""):
#     plt.axes(ax)
#     ax.cla()
#     ax.bar(e, z, width=e[1]-e[0], bottom=0.3)
#     ax.plot(s['direction'], s['radius'], 'ro')
#     ax.plot([t['direction']]*2, [0, t['radius']], 'g-', linewidth=5)
#     ax.set_ylim(0,0.5)
#     ax.set_title(title)
#     plt.box(False)
#     plt.yticks([], [])
#     plt.draw()

t = ime_samps['direction']
r = ime_samps['radius']

tt = truth['direction']
tr = truth['radius']

x = np.cos(t)*r
y = np.sin(t)*r

tx = np.cos(tt)*tr
ty = np.sin(tt)*tr

# x = np.empty((1, 1, 48))
# x[:, :, :24] = 0.2205
# x[:, :, 24:] = -0.2205
# y = np.empty((1, 1, 48))
# y[:, :, :24] = -0.2205
# y[:, :, 24:] = 0.2205

# tx = np.zeros((1, 1))
# ty = np.zeros((1, 1))

data = np.concatenate([x[..., None], y[..., None]], axis=-1)

n = (20, 20)
edges, binsize = kde.gen_xy_edges(n)
mids = (edges[:, 1:] + edges[:, :-1]) / 2.

sclx = lambda x: (((x - edges[0,0]) / (edges[0,-1]-edges[0,0])) * (edges.shape[1]-1)) - 0.5
scly = lambda y: (((y - edges[1,0]) / (edges[1,-1]-edges[1,0])) * (edges.shape[1]-1)) - 0.5

for i in xrange(data.shape[0]):
    bx = 1.3**kde.xy_kde(data[i], n, h=0.2, s=-.35, t=.35)
    for j in xrange(data.shape[1]):
        title = "r=%.1f" % ratios[j]
        ax = axes.ravel()[j]
        plt.axes(ax)
        ax.cla()
        ax.imshow(bx[j].T, interpolation='nearest', vmin=0, vmax=1)
        ax.plot(sclx(x[i,j]), scly(y[i,j]), 'ro')
        ax.plot(sclx(tx[i,j]), scly(ty[i,j]), 'yo')
        ax.set_xticks(np.arange(len(edges[0]))-0.5)
        ax.set_xticklabels(edges[0])
        ax.set_yticks((np.arange(len(edges[0]))-0.5))
        ax.set_yticklabels(edges[1])
        ax.set_title(title)
        plt.draw()
        # hist(axes.ravel()[j],
        #      edges[:-1],
        #      np.exp(p_outcomes[i,j]),
        #      ime_samps[i,j],
        #      truth[i,j],
        #      title=title)
        print title#, model_subjects[j,i]
    pdb.set_trace()


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
        # p_obs_t = p_obs[:, truth_t]
        # obs_t = weightedSample(
        #     np.exp(p_obs_t), n_part, axis=0, rso=rso).T

        # compute responses
        m_response = mo.response(thetas_t, p_outcomes_t, loss)
        m_lh, m_joint, m_theta = mo.learningCurve(
            obs_t, thetas_t, p_outcomes_t)

        # calculate weights
        w = np.ones((n_part, n_responses))
        w[m_response[:, [0]] == np.arange(n_responses)] += B
        p = np.log(w / np.expand_dims(np.sum(w, axis=-1), axis=-1))
        weights_t = normalize(p[:, response_t])[1]

        # sample new particles
        tidx = weightedSample(
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
