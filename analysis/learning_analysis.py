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
        hd = htrial[hidx]
        ho = horder[hidx]
        
        sort = np.argsort(ho)
        shuman = hd[sort]
        sorder = ho[sort]
        sstim = stimuli[..., sidx, :, :][..., sort, :, :]
        smodel = model[..., sidx, :, :][..., sort, :, :]
        
        trial_human.append(shuman)
        trial_order.append(sort)
        trial_stim.append(sstim)
        trial_model.append(smodel)
        
    trial_human = np.array(trial_human)
    trial_order = np.array(trial_order)
    trial_stim = np.array(trial_stim)
    trial_model = np.array(trial_model)

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
pfell, nfell, fell_sample = tat.process_model_stability(
    rawmodel, mthresh=mthresh, zscore=zscore, pairs=False)

all_model = np.array([truth_samples, fell_sample])
fellsamp = all_model[:, 0, 0].transpose((0, 2, 1, 3)).astype('int')

assert (rawhstim == rawsstim).all()

human, stimuli, sort, model = order_by_trial(rawhuman, rawhstim, raworder, fellsamp)
kappa10 = list(kappas).index(1)

######################################################################

class ModelObserver(object):

    n_F = 11
    n_R = 7

    OUTCOMES = np.arange(n_F)
    RESPONSES = np.arange(n_R)

    def __init__(self, ime_samples, kappas, N=1, Cf=10, Cr=6):

        self.ime_samples = ime_samples.copy()

        self.kappas = kappas.copy()
        self.n_kappa = len(self.kappas)
        self.thetas = [np.ones(self.n_kappa) / float(self.n_kappa)]

        self.time = 0
        self.stimuli = []

        self._N = N
        self._Cf = Cf
        self._Cr = Cr
        self._loss = None
        self._risks = []

        self._P_F_kappas = []
        self._P_Fts = []
        self._Pt_kappas = []

    def IME(self):
        samps = self.ime_samples[self.time]
        shape = list(samps.shape[:-1]) + [n_F]
        succ = samps[:, None, :] == self.OUTCOMES[None, :, None]
        prior = rvs.Dirichlet(np.ones(shape))
        post = prior.updateMultinomial(succ, axis=-1)
        mle = post.mean.copy()
        return mle

    @property
    def P_Ft_kappa(self):
        """P(F_t | S_t, kappa)"""
        
        try:
            dist = self._P_F_kappas[self.time]

        except IndexError:
            # Get the probability of outcomes from the IME
            p_IME = self.IME()

            # Make bernoulli distributions with these probabilities as
            # parameters, and evaluate the likelihood of the true
            # outcome
            dist = rvs.Multinomial(np.ones(p_IME.shape), p_IME)
            self._P_F_kappas.append(dist)
        
        return self._P_F_kappas[self.time]

    @property
    def Pt_kappa(self):
        """P_t(kappa) = theta_t"""

        try:
            dist = self._Pt_kappas[self.time]

        except:
            p = self.thetas[self.time]
            dist = rvs.Multinomial(np.ones(p.shape), p)
            self._Pt_kappas.append(dist)
            
        return self._Pt_kappas[self.time]

    @property
    def P_Ft(self):
        """P(F_t | S_t)"""

        try:
            dist = self._P_Fts[self.time]

        except IndexError:
            p_F_kappa = np.log(self.P_Ft_kappa.p)
            Pt_kappa = np.log(self.Pt_kappa.p)[:, None]
            p_F = stats.normalize(p_F_kappa + Pt_kappa, axis=0)[0]
            dist = rvs.Multinomial(np.ones(p_F.shape), np.exp(p_F))
            self._P_Fts.append(dist)

        return self._P_Fts[self.time]
        

    @property
    def Loss(self):
        if self._loss is None:
            N, Cf, Cr = self._N, self._Cf, self._Cr
            f = self.OUTCOMES[:, None]
            r = self.RESPONSES[None, :]
            
            sf = ((f + N).astype('f8') / n_F) - 0.5
            sr = ((r - N).astype('f8') / n_R) - 0.5
            ssf = 1.0 / (1 + np.exp(-Cf * sf))
            ssr = 1.0 / (1 + np.exp(-Cr * sr))
            self._loss = np.sqrt(np.abs(ssf - ssr))

        return self._loss
    
    def Risk(self):
        try:
            risk = self._risks[self.time]

        except IndexError:
            p_F = self.P_Ft.p[:, None]
            loss = self.Loss
            risk = np.sum(loss * p_F, axis=0)
            self._risks.append(risk)
            
        return risk

    ##################################################################

    def viewStimulus(self, S):
        self.stimuli.append(S)
        self.time = len(self.stimuli) - 1

    def generateResponse(self):
        """Compute optimal response to the current stimulus"""
        response = np.argmin(self.Risk(), axis=0)
        return response

    def viewFeedback(self, F):
        # Likelihood of the true outcomes
        lh_F_kappa = self.P_Ft_kappa.logPMF(F == self.OUTCOMES)
        p_kappa = np.log(self.Pt_kappa.p)

        ## Calculate theta_t = P_t(kappa)
        joint = lh_F_kappa + p_kappa
        Pt_kappa = stats.normalize(joint, axis=-1)[1]
        self.thetas.append(np.exp(Pt_kappa))
        self.time += 1

        return np.exp(lh_F_kappa)

R = 6 - human.copy()
S = stimuli[0].copy()
F = model[0, 0, :, kappa10, 0].copy()

assert R.shape[1:] == S.shape
assert S.shape == F.shape

n_subj = R.shape[0]
n_T = R.shape[1]
n_F = 11
n_R = 7
n_kappa = len(kappas)
ratios = np.round(10 ** kappas, decimals=1)

MO = ModelObserver(
    ime_samples=model[0, 1].copy(),
    kappas=kappas,
    N=1,
    Cf=10,
    Cr=6)

R = []
Pt_kappa = []

plt.close('all')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

for t in xrange(n_T):

    MO.viewStimulus(S[t])
    Pt_kappa.append(MO.Pt_kappa.p.copy())
    R.append(MO.generateResponse())
    lh_F_kappa = MO.viewFeedback(F[t])

    print "Trial %d" % t
    print "\tResponse = %d" % R[t]
    print "\tFeedback = %d" % F[t]

    # ax1.cla()
    # ax1.set_xticks(np.arange(n_kappa))
    # ax1.set_xticklabels(ratios)
    # ax1.set_ylabel("P(kappa)")
    # ax1.bar(np.arange(n_kappa), Pt_kappa[t], align='center')
    # ax1.set_ylim(0, 1)
    # ax1.set_title("Trial %s" % t)

    # ax2.cla()
    # ax2.set_xticks(np.arange(n_kappa))
    # ax2.set_xticklabels(ratios)
    # ax2.set_ylabel("LH(F=%d | kappa)" % F[t])
    # ax2.bar(np.arange(n_kappa), lh_F_kappa, align='center')
    # ax2.set_ylim(0, 1)

    # ax3.cla()
    # ax3.set_xticks(np.arange(n_kappa))
    # ax3.set_xticklabels(ratios)
    # ax3.set_ylabel("P(kappa | F)")
    # ax3.bar(np.arange(n_kappa), MO.Pt_kappa.p, align='center')
    # ax3.set_ylim(0, 1)

    # ax4.cla()
    # ax4.set_xticks(np.arange(n_F))
    # ax4.set_xticklabels(np.arange(n_F))
    # ax4.set_ylabel("P(F)")
    # ax4.bar(np.arange(n_F), MO.P_Ft.p, align='center')
    # ax4.set_ylim(0, 1)

    # plt.draw()
    # plt.draw()

    # time.sleep(0.05)
    # #pdb.set_trace()

Pt_kappa = np.array(Pt_kappa)

plt.figure()
plt.imshow(
    np.log(Pt_kappa.T),
    aspect='auto', interpolation='nearest',
    vmin=-250, vmax=0)
plt.xlabel("Trial Number")
plt.ylabel("Mass Ratio")
plt.yticks(np.arange(n_kappa), ratios)
plt.title("model observer")






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
