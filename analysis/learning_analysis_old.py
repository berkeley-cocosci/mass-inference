import numpy as np
import scipy.stats
import pdb
from matplotlib import rc
from sklearn import linear_model
import matplotlib.cm as cm

#import cogphysics.lib.dataio as dio
from cogphysics.lib.corr import xcorr
import cogphysics.lib.circ as circ
import cogphysics.lib.nplib as npl
import cogphysics.lib.rvs as rvs
import cogphysics.lib.stats as stats

import cogphysics.tower.analysis_tools as at

rc('font', family='serif')
rc('text', usetex=True)

def load_model(name):

    # load model predictions
    arr, meta = dio.load(name)

    kappas = meta['dimvals']['kappa']
    sigmas = meta['dimvals']['sigma']
    mus = meta['dimvals']['mu']

    # shape variables
    numkappa = arr.shape[0]
    numsigma = arr.shape[1]
    nummu    = arr.shape[2]
    numstim  = arr.shape[3]
    numsamp  = arr.shape[4]
    
    # calculate stability
    midx = mus.index(0.8)
    posdiff = np.sum(
        (arr[:, :, midx, ..., -1, :3] - arr[:, :, midx, ..., 2, :3]) ** 2,
        axis=-1) ** 0.5
    moved = posdiff > 0.1
    pfell = (np.sum(moved, axis=-1) > 4).astype('f8')
    nfell = np.sum(moved, axis=-1).astype('int')

    # calculate direction
    assign = np.array([[int(y) for y in x.split("_")[-1]]
                       for x in meta['dimvals']['cpo']])
    mass = assign[None, None, :, None, :]
    mass = mass * (10**np.array(kappas, dtype='f8')[:, None, None, None, None])
    mass = mass * np.ones(moved.shape)
    mass[mass==0] = 1

    movedweight = mass[..., None] * np.ones(moved.shape)[..., None]
    movedweight[~moved] = np.nan
    movedweight = movedweight / np.nansum(movedweight, axis=-2)[..., None, :]

    allweight = mass[..., None] * np.ones(moved.shape)[..., None]
    allweight = allweight / np.nansum(allweight, axis=-2)[..., None, :]
    
    com0 = np.nansum(arr[:, :, midx, ..., 2, :2] * allweight, axis=-2)
    comt = np.nansum(arr[:, :, midx, ..., -1, :2] * movedweight, axis=-2)
    comdiff = comt - com0
    direction = np.arctan2(comdiff[..., 1], comdiff[..., 0])

    hans_pfell = pfell[:, sigmas.index('0.04')]
    hans_nfell = nfell[:, sigmas.index('0.04')]
    hans_dir = direction[:, sigmas.index('0.04')]
    truefell = nfell[kappas.index('1.0'), sigmas.index('0.00'), :, 0] > 0
    truedir = direction[kappas.index('1.0'), sigmas.index('0.00'), :, 0]

    stims = meta['dimvals']['cpo']
    kappas = [float(x) for x in meta['dimvals']['kappa']]

    return hans_pfell, hans_nfell, truefell, hans_dir, truedir, stims, kappas

def load_human(name):

    # load human data
    harr, hmeta = dio.load(name)

    # shape variables
    numsubj = harr.shape[0]
    numstim = harr.shape[1]
    numrep = harr.shape[2]

    # reshape the human trials so they're in order of the stimuli that people
    # saw
    harr2 = harr.reshape((numsubj, numstim*numrep))
    trialsort = np.argsort(harr2['current_trial'], axis=-1)
    harrsort = np.vstack([harr2[i, trialsort[i]][None] for i in xrange(numsubj)])

    # get human responses and stimuli names
    hans = harrsort['answer'] / 6.
    stims = np.array([[y.split("~")[0] for y in x] for x in harrsort['stimulus']])
    
    return hans, stims

def load_cues(name):
    # load cues
    carr, cmeta = dio.load(name)
    cnames = list(carr.dtype.names)
    carr = carr.view('f8').reshape((carr.shape[0], carr.shape[1], -1))
    carr = carr.transpose((0, 2, 1))[0]

    dircues = ['light_skew_dir', 'heavy_skew_dir']
    cdirnames = cnames[:]
    cdirarr = carr.copy()
    for cue in cnames:
        if cue not in dircues:
            cdirarr = np.delete(cdirarr, cdirnames.index(cue), axis=0)
            cdirnames.remove(cue)

    for cue in cnames[:]:
        if cue.startswith("all") or cue.endswith("dir"):
            carr = np.delete(carr, cnames.index(cue), axis=0)
            cnames.remove(cue)
            
    return carr, cdirarr, cnames, cdirnames

################################################################################

# def model_direction(vals, truevals, trial_order):
#     ntrial = trial_order.size
#     nkappa = vals.shape[0]

#     # MAP probability of direction
#     vm = rvs.VonMises.MAP(vals, axis=-1, nanrobust=True)

#     # compute probability of kappa given observed direction
#     p_F = vm.logpdf(truevals[None]).T[trial_order]
#     p_F[np.isnan(truevals[trial_order])] = 0
#     pkappa = np.zeros((ntrial+1, nkappa))
#     pkappa[0, :] = np.log(np.ones(nkappa) / nkappa)
#     for itime in xrange(ntrial):
#         pkappa[itime+1] = fl.normalize((pkappa[itime] + p_F[itime]))[1]

#     return pkappa

# def heuristic_direction(vals, truevals, trial_order):
#     pass

def stability_modelObserver(vals, truevals, hans, mtrial_order, stim_order):

    vals = 1-pfell.copy()
    truevals = 1-truefell.copy()
    hans = stim_hpred.copy()

    ntrial = mtrial_order.size
    nkappa = vals.shape[0]
    nsubj = hans.shape[0]

    alpha = 1

    htrial_order = np.argsort(stim_order)
    mo_samples = vals[:, mtrial_order, :]
    mo_truth = truevals[mtrial_order]
    subj_response = hans.reshape((nsubj, ntrial))[:, htrial_order] * 6
    subj_response_mn = (
        subj_response[..., None] == np.arange(7)[None, None, :]).astype('i8')


    # beta posterior for the binomial parameter, which specifies the probability
    # of falling (for the model), using a Jeffreys prior
    beta_prior = rvs.Beta(0.5, 0.5)
    beta_posterior = beta_prior.updateBernoulli(mo_samples, axis=-1)

    # log probability of falling, according to the model
    P_fell = beta_posterior.mode
    P_fell[np.isnan(P_fell)] = beta_posterior.mean[np.isnan(P_fell)]
    P_fell = np.log(P_fell)
    # posterior over stability (true/false), according to the model
    binom_posterior = rvs.Binomial(1, np.exp(P_fell))

    # find the weights of the 7 choices from the CDF of the beta posterior of
    # falling
    choices = np.linspace(0, 1, 8)[1:]
    cweights = beta_posterior.CDF(choices[:, None, None]).transpose((1, 2, 0))
    cweights[..., 1:] = cweights[..., 1:] - cweights[..., :-1]
    cweights = cweights ** 1
    # the distribution over responses is a multinomial parameterized by these
    # weights
    response_dist = rvs.Multinomial(1, cweights, axis=-1)

    # the log likelihood of each possible response
    L_response = response_dist.logPMF(np.eye(7)[:, None, None, :])
    # log likelihood of true outcome given kappa, according to the model
    L_truth = binom_posterior.logPMF(mo_truth)

    # log probability of kappa given observed stability
    P_kappa = np.zeros((nkappa, ntrial+1))
    P_kappa.fill(-np.inf)
    P_kappa[:, 0] = np.log(np.ones(nkappa) / nkappa)
    for itime in xrange(ntrial):
        P_kappa[:, itime+1] = fl.normalize(
            P_kappa[:, itime] + L_truth[:, itime], axis=0)[1]

    # log probability of kappa given observed stability
    P_response = np.sum(np.exp(
        P_kappa[None, :, :-1] + L_response), axis=1)
    mo_response = np.argmax(P_response, axis=0).astype('f8') / 6.

    print xcorr(
        np.mean(mo_response[stim_order].reshape((nstim, -1)), axis=-1),
        np.mean(subj_response[:, stim_order].reshape(
            (nsubj, nstim, -1)), axis=-1))

    return P_kappa, mo_response

def stability_experimenter(vals, truevals, responses, mtrial_order, stim_order):

    vals = 1-pfell.copy()
    truevals = 1-truefell.copy()
    htrial_order = np.argsort(stim_order)
    mo_samples = vals[:, mtrial_order, :]
    mo_truth = truevals[mtrial_order]
    subj_response = hans.reshape((nsubj, ntrial))[:, htrial_order] * 6
    subj_response_mn = (
        subj_response[..., None] == np.arange(7)[None, None, :]).astype('i8')
    mo_response_mn = (mo_response[:, None]*6 == np.arange(7)[None, :]).astype('i8') 
    responses = np.vstack((subj_response_mn, mo_response_mn[None]))

    ntrial = mtrial_order.size
    nkappa = vals.shape[0]
    nsubj = hans.shape[0]

    alpha = 1

    # dirichlet posterior over the probability of responses for the model, again
    # using a Jeffreys prior
    dir_prior = rvs.Dirichlet(np.ones(choices.shape)*0.5)
    dir_posterior = dir_prior.updateMultinomial(
        np.round(cweights[..., None]*100), axis=-1)

    # log probability of responses, according to the model
    P_response = dir_posterior.mode
    P_response[np.isnan(P_response)] = dir_posterior.mean[np.isnan(P_response)]
    P_response /= np.sum(P_response, axis=-1)[..., None]
    P_response = np.log(P_response)
    # posterior over responses (0-6), according to the model
    mn_posterior = rvs.Multinomial(1, np.exp(P_response))

    # log likelihood of human and model responses under the model's multinomial
    # posterior
    L_response = mn_posterior.logPMF(responses[:, None, :, :])

    # log probability of theta given observed responses
    P_theta = np.zeros((nsubj+1, nkappa, ntrial+1))
    P_theta.fill(-np.inf)
    P_theta[..., 0] = np.log(np.ones((nsubj+1, nkappa)) / nkappa)
    for itime in xrange(ntrial):
        P_theta[..., itime+1] = fl.normalize(
            P_theta[..., itime] + L_response[..., itime])[1]


        
    plt.figure()
    for i in xrange(nsubj+1):
        plt.subplot(1, nsubj+2, i+1)
        plt.imshow(P_theta[i], aspect='auto',
                   interpolation='nearest', vmin=-300, vmax=0)
        plt.xlabel("Trial Number")
        if i == 0:
            plt.ylabel("Mass Ratio")
            plt.yticks(np.arange(nkappa), np.round(10**np.array(kappas), decimals=1))
        else:
            plt.yticks([], [])
        title = "subject %d" % (i+1) if i < nsubj else "model observer (inferred)"
        plt.title(title)
    plt.subplot(1, nsubj+2, nsubj+2)
    plt.imshow(P_kappa, aspect='auto',
               interpolation='nearest', vmin=-300, vmax=0)
    plt.xlabel("Trial Number")
    plt.yticks([], [])
    plt.title("model observer")
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)

    return pkappa, preds

# def heuristic_stability(vals, truevals, hans, torder, sorder):

#     ncue = vals.shape[0]
#     ntrial = torder.size
    
#     clf = linear_model.RidgeCV()
#     clf.intercept_ = 0
#     feats = vals.T[torder]
#     true = truevals[torder]

#     coeffs = np.zeros((ntrial+1, ncue))
#     preds = np.zeros(ntrial)

#     coeffs[0, :] = np.ones(ncue) / ncue
#     clf.coef_ = coeffs[0, :].copy()
#     preds[0] = clf.predict(feats[[0]])
    
#     coeffs[1, :] = np.ones(ncue) / ncue
#     clf.coef_ = coeffs[1, :].copy()
#     preds[1] = clf.predict(feats[[1]])
    
#     for itime in xrange(1, ntrial):
#         clf.fit(feats[:itime+1], true[:itime+1])
#         coeffs[itime+1] = clf.coef_
#         if itime+1 < ntrial:
#             preds[itime+1] = clf.predict(feats[[itime+1]])

#     ckappa = np.log10(coeffs[:, 4] / coeffs[:, 3])
#     stim_cpred = preds[sorder].reshape((nstim, -1))
#     print xcorr(
#         np.mean(stim_cpred, axis=-1),
#         np.mean(hans, axis=-1))

#     return coeffs, preds
        
################################################################################

# load model samples and human judgments
names = [
    # ("s_predict-mass-towers",
    #  "h_predict-stability-towers_experiment~kappa-1.0",
    #  "s_mass-towers_heuristics"),
    
    ("s_predict-mass-new-stability-towers",
     "h_predict-new-stability-towers_experiment~kappa-1.0",
     "s_mass-new-stability-towers_heuristics"),

    # ("s_predict-mass-direction-towers",
    #  "h_predict-direction-towers_experiment~kappa-1.0",
    #  "s_mass-direction-towers_heuristics"),

    # ("s_predict-mass-new-direction-towers",
    #  "h_predict-new-direction-towers_experiment~kappa-1.0",
    #  "s_mass-new-direction-towers_heuristics"),

    ]

sname, hname, cname = names[0] 

for sname, hname, cname in names:

    pfell, nfell, truefell, dir, truedir, mstim, kappas = load_model(sname)
    cfell, cdir, fellcues, dircues = load_cues(cname)

    nstim = pfell.shape[1]

    # stimuli ordering
    hans, hstim = load_human(hname)
    mtrial_order = np.array([mstim.index(i) for i in hstim[0]])
    stim_order = np.argsort(mtrial_order)
    stim_hpred = hans[:, stim_order].reshape((hans.shape[0], nstim, -1))

    # constants
    ntrial = len(mtrial_order)
    nkappa = len(kappas)
    nstim = len(mstim)
    nfellcue = len(fellcues)
    ndircue = len(dircues)

    # if "direction" in sname:
    #     model = model_direction(dir, truedir, mtrial_order)

    #     plt.figure()
    #     plt.imshow(pkappa.T, aspect='auto', interpolation='nearest', vmin=-300, vmax=0)
    #     plt.xlabel("Trial Number")
    #     plt.ylabel("Mass Ratio")
    #     plt.yticks(np.arange(nkappa), np.round(10**np.array(kappas), decimals=1))
    #     plt.title("%s (direction)" % sname.lstrip("s_").replace("_", " ").replace("-", " "))

    # else:

    #     mprob, mpred = model_stability(
    #         1-pfell, 1-truefell, stim_hpred, mtrial_order, stim_order)
    #     ccoeff, cpred = heuristic_stability(
    #         cfell, 1-truefell, stim_hpred, mtrial_order, stim_order)
        
    #     plt.figure()
    #     plt.imshow(mprob.T, aspect='auto', interpolation='nearest', vmin=-300, vmax=0)
    #     plt.xlabel("Trial Number")
    #     plt.ylabel("Mass Ratio")
    #     plt.yticks(np.arange(nkappa), np.round(10**np.array(kappas), decimals=1))
    #     plt.title("%s (stability)" % sname.lstrip("s_").replace("_", " ").replace("-", " "))

    #     plt.figure()
    #     plt.imshow(ccoeff.T, aspect='auto', interpolation='nearest', vmin=-2, vmax=2)
    #     plt.xlabel("Trial Number")
    #     plt.ylabel("Heuristic")
    #     plt.yticks(np.arange(nfellcue), [x.replace("_", " ") for x in fellcues])
    #     plt.title("%s (stability)" % cname.lstrip("s_").replace("_", " ").replace("-", " "))





















def L(F, R):
    # F is in {0, 1}
    # R is in {0, 1, 2, 3, 4, 5, 6}
    r = R / 6.
    #if F == 0:
    C = r ** 3
    #elif F == 1:
    #    C = - (r ** 2)
    risk = (F - r) - C
    return risk
F = np.linspace(0, 1, 11)
R = np.arange(7)
EU = (L(0, R)[:, None] * pF[None]) + (L(1, R)[:, None] * (1 - pF[None]))
plt.figure(0)
plt.clf()
plt.plot(R, L(0, R), 'b'); plt.plot(R, L(1, R), 'r')
plt.figure(1)
plt.clf()
plt.imshow(EU, cmap=cm.gray, interpolation='nearest')
plt.xticks(np.arange(11), np.round(pF, decimals=1))
plt.yticks(R, R)
