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

from cogphysics.lib.corr import xcorr

import model_observer as mo
import learning_analysis_tools as lat

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))

######################################################################
## Load and process data

out = lat.load('stability')
rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = out
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

human, stimuli, sort, truth, ipe = lat.order_by_trial(
    rawhuman, rawhstim, raworder, rawtruth, rawipe)
truth = truth[0]
ipe = ipe[0]
 
# variables
n_trial      = stimuli.shape[1]
n_kappas     = len(kappas)

######################################################################
# Model observer for each true mass ratio

outcomes = np.array([0, 1])
responses = np.array([0, 1])
loss = mo.Loss(outcomes, responses)

nthresh0, nthresh, nsamps = 1, 4, 300
smooth, decay = True, 1.0
feedback, ipe_samps = lat.make_observer_data(
    nthresh0, nthresh, nsamps)
lh, joint, p_kappas, responses = mo.ModelObserver(
    ipe_samps,
    feedback[:, None],
    outcomes=outcomes,
    loss=loss,
    smooth=smooth)

h_responses = (human > 4).astype('f8')
h_responses[human == 4] = np.nan

# true_responses = responses.copy().astype('i8')
# true_responses = mo.responses(
#     p_kappas[:, [-1]]*np.ones(p_kappas.shape),
#     outcomes[:, None], ipe_samps, loss, smooth)
# sidxs = [0, 3, 6, 8, 10, 13, 16, 18, 20, 23, 26]

true_responses = h_responses.copy()
sidxs = np.arange(true_responses.shape[0])
n_subj = len(sidxs)
n_part = 200
n_responses = 2
B = 1
rand = 0.5
kappa0 = np.nonzero(kappas==1.0)[0][0]
neffthresh = n_part * .75

# arrays to hold particles and weights
thetas = np.empty((n_subj, n_part, n_trial+1, n_kappas)) * np.nan
weights = np.empty((n_subj, n_part, n_trial+1)) * np.nan
mle_alphas = np.empty((n_subj, n_trial+1, n_kappas)) * np.nan

# initial particle values and weights
rso = np.random.RandomState(50)
def make_prior(alpha=5, ell=0.2, eps=0.0005):
    kernel = mo.make_rbf_kernel(alpha=alpha, ell=ell)
    K = kernel(kappas[:, None], kappas[None, :]) + (eps * np.eye(n_kappas))
    def draw(n):
        f = rso.multivariate_normal(np.zeros(n_kappas), K, n)
        sf = normalize(np.log(1. / (1. + np.exp(-f))), axis=-1)[1]
        return sf
    return draw
prior = make_prior()
sf = prior((n_subj, n_part))
plt.figure(2)
plt.clf()
plt.plot(np.exp(sf[0]).T)
plt.ylim(0, 1)
thetas[:, :, 0] = sf
weights[:, :, 0] = np.log(np.ones((n_subj, n_part)) / n_part)
mle_alphas[:, 0] = np.ones((n_subj, n_kappas)) / n_kappas

for idx, sidx in enumerate(sidxs):

    print idx
    rso.seed(100)
        
    for t in xrange(0, n_trial):

        thetas_t = thetas[idx, :, t].copy()#[:, None]
        weights_t = weights[idx, :, t].copy()
        samps_t = ipe_samps[t]
        #obs_t = feedback[t, sidx]
        obs_t = feedback[t, kappa0]
        response_t = true_responses[sidx, t]

        if not np.isnan(response_t):
            response_t = int(response_t)
            
            # compute responses
            tm_response = mo.response(
                thetas_t, outcomes[:, None], samps_t, loss, smooth)
            # rand = np.empty(tm_response.shape)
            # rand[tm_response==1] = pf
            # rand[tm_response==0] = pnf
            #r = (rso.uniform(0, 1, tm_response.shape) < rand).astype('i8')
            #m_response = (tm_response + r) % 2
            m_response = tm_response.copy()

            f = mo.IPE(samps_t, smooth=smooth)
            ef = mo.evaluateFeedback(obs_t, f)
            noise = prior(n_part)
            #noise = np.zeros(thetas_t.shape)
            m_theta = normalize(thetas_t + ef + noise, axis=-1)[1]

            # calculate weights
            w = np.zeros((n_part, n_responses)) + rand
            w[m_response[:, None] == np.arange(n_responses)] += B
            p = np.log(w / np.expand_dims(np.sum(w, axis=-1), axis=-1))
            weights_t = normalize(p[:, response_t] + weights_t)[1]

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

msubj_responses = mo.responses(
    np.log(np.mean(np.exp(thetas), axis=1)),
    outcomes[:, None],
    ipe_samps,
    loss,
    smooth=smooth)

fig = plt.figure(3)
plt.clf()
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
    err0 = scipy.stats.nanmean(np.abs(true_responses[i] - responses[23]))
    err1 = scipy.stats.nanmean(np.abs(true_responses[i] - msubj_responses[i]))
    print "%.2f --> %.2f (% .2f)" % (err0, err1, err1-err0)

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
