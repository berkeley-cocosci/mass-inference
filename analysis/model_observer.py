from __future__ import division
import numpy as np
from snippets.safemath import normalize


def make_kde_smoother(x, lam):
    dists = np.abs(x[:, None] - x[None, :]) / lam
    pdist = np.exp(-0.5 * dists**2) / np.sqrt(2)
    sum_pdist = np.sum(pdist, axis=-1)

    def kde_smoother(y):
        est = np.sum(pdist * y, axis=-1) / sum_pdist
        return est
    return kde_smoother


def binom_mean(samps, axis=None):
    alpha = np.sum(samps, axis=axis) + 0.5
    beta = np.sum(1-samps, axis=axis) + 0.5
    pfell_mean = alpha / (alpha + beta)
    return pfell_mean


def binom_std(samps, axis=None):
    alpha = np.sum(samps, axis=axis) + 0.5
    beta = np.sum(1-samps, axis=axis) + 0.5
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    return pfell_std


def IPE(F, samps, kappas, smooth):
    """Evaluate the posterior probability of the outcome given
    kappa using samples from the internal mechanics engine.

    P(F | S, k) = P(k | F, S) P(F)

    Parameters
    ----------
    F : array-like (..., n_trial)
        outcomes
    samps : array-like (n_trial, n_kappas, n_samples)
        samples from the IPE
    smooth : boolean
        whether to smooth the IPE estimates

    Returns
    -------
        P(F | S, k) with shape (..., n_trial, n_kappas)

    """

    n_trial, n_kappas, n_samps = samps.shape
    x = np.array(kappas)

    # shape is (n_trial, n_kappas)
    pfell_mean = binom_mean(samps, axis=-1)
    # pfell_meanstd = np.mean(pfell_std, axis=-2)

    if not smooth:
        # using bernoulli fall/not fall
        pfell = pfell_mean

    else:
        # using bernoulli fall/not fall plus kernel smoothing
        #lam = (pfell_meanstd * 10)
        lam = 0.2
        # calculate distances
        # shape is (n_kappas, n_kappas)
        dists = np.abs(x[:, None] - x[None, :]) / lam
        # calculate gaussian probability of distances
        pdist = np.exp(-0.5 * dists**2) / np.sqrt(2)
        # shape is (n_kappas,)
        sum_pdist = np.sum(pdist, axis=-1)
        # broadcasts to shape (n_trial, n_kappas, n_kappas), then sum
        # leads to shape of (n_trial, n_kappas)
        pfell = np.sum(pdist * pfell_mean[..., None, :], axis=-1) / sum_pdist

    # Ok, so now we have pfell which is shape (n_trial, n_kappas) and
    # is the probability of the stimulus on trial N falling, given
    # kappa. So now we need to actually evaluate that probability...

    # likelihood of falling = pfell
    lh1 = F[..., None] * pfell
    # likelihood of not falling = 1-pfell
    lh0 = (1-F)[..., None] * (1-pfell)
    # shape is going to be (..., n_trial, n_kappas), where ... refers
    # to whatever the rest of the shape of F was
    lh = lh0 + lh1
    #pdf = np.swapaxes((lh1 + lh0)[..., 0, 0], -2, -1)
    return lh


def predict(p_kappas, ipe_samps, kappas, smooth):
    """Predict the likelihood of the outcome given the stimulus, P(F_t
    | S_t), independent of kappa.

    Parameters
    ----------
    p_kappas : array-like (..., n_trial, n_kappas)
        prior, log theta_t = log P_t(kappa)
    ipe_samps : array-like (n_trial, n_kappas, n_samples)
        samples from the IPE
    smooth : boolean
        whether to smooth the IPE estimates

    """

    n_trial, n_kappas, n_samples = ipe_samps.shape

    # get the log likelihood, P(F | S, k), in the shape of (n_trial,
    # n_kappas)
    llh = np.log(IPE(np.ones(n_trial), ipe_samps, kappas, smooth))
    llh[np.isnan(llh)] = np.log(0.5)
    # compute the joint probability, P(F, k | S)
    joint = llh + p_kappas
    # marginalize out kappa, P(F | S), shape is now (..., n_trial),
    # where ... refers to whatever rest of the shape of p_kappas was
    p_outcomes = normalize(joint, axis=-1)[0]
    return p_outcomes


def ModelObserver(feedback, ipe_samps, kappas,
                  prior=None, p_ignore=0.0, smooth=True):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    prior over mass ratios, and the probability of each possible
    outcome.

    P_t(k) = P(F_t | S, k) P_t-1(k)

    Parameters
    ----------
    feedback : array-like (..., n_trial)
        True outcomes of each trial
    ipe_samps : array-like (n_trial, n_kappas, n_samples)
        samples from the IPE
    smooth : boolean
        whether to smooth the IPE estimates

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       joints : np.ndarray (n_trial+1, n_kappas)
       thetas : np.ndarray (n_trial+1, n_kappas)

    """

    # n_trial = feedback.shape[0]
    # n_kappas = ipe_samps.shape[-3]
    n_trial, n_kappas, n_samps = ipe_samps.shape

    # prior, if none is given, shape of (n_kappas,)
    if prior is None:
        prior = np.log(np.ones(n_kappas) / n_kappas)

    # get the log likelihood, P(F | S, k), in the shape of (...,
    # n_trial, n_kappas)
    lh = np.log(IPE(feedback, ipe_samps, kappas, smooth))
    lh[np.isnan(lh)] = np.log(0.5)

    # construct a sequence of the prior and all the likelihoods
    seq = np.empty(lh.shape[:-2] + (n_trial+1, n_kappas))
    seq[..., 0, :] = prior
    seq[..., 1:, :] = lh
    # cumulative sum over this sequence to get the joint P(F, k | S)
    joint = np.cumsum(seq, axis=-2)
    # normalize to get the posterior P(k | F, S)
    posterior = normalize(joint, axis=-1)[1]

    return joint, posterior


def EvaluateObserverTrials(responses, feedback, ipe_samps, kappas,
                           prior=None, p_ignore=0.0, smooth=True):

    joint, posterior = ModelObserver(
        feedback, ipe_samps, kappas,
        prior=prior, p_ignore=p_ignore, smooth=smooth)

    n_trial, n_kappas, n_samps = ipe_samps.shape
    lh = np.log(IPE(np.ones(n_trial), ipe_samps, kappas, smooth))
    lh[np.isnan(lh)] = np.log(0.5)

    p_fall = np.exp(normalize(posterior[..., :-1, :] + lh, axis=-1)[0])
    p_response = (p_fall * (1-p_ignore)) + ((1-p_fall) * (p_ignore))

    # marginal and responses should be in the shape (..., n_trial)
    lh0 = responses * p_response
    lh1 = (1-responses) * (1-p_response)
    lh = lh0 + lh1
    lh[np.isnan(lh)] = 0.5
    assert (lh <= 1).all()
    return np.log(lh)


def EvaluateObserver(responses, feedback, ipe_samps, kappas,
                     prior=None, p_ignore=0.0, smooth=True):

    lh = EvaluateObserverTrials(
        responses, feedback, ipe_samps, kappas,
        prior=prior, p_ignore=p_ignore, smooth=smooth)
    return np.sum(lh, axis=-1)


def simulateResponses(n, feedback, ipe_samps, kappas,
                      prior=None, p_ignore=0.0, smooth=True, rso=None):

    # learning model beliefs
    model_joint, model_theta = ModelObserver(
        feedback, ipe_samps, kappas, prior=prior, smooth=smooth)

    # compute probability of falling
    p_fall = np.exp(predict(
        model_theta[..., :-1, :], ipe_samps, kappas, smooth))
    p_response = (p_fall * (1-p_ignore)) + ((1-p_fall) * (p_ignore))

    # sample responses
    n_trial, n_kappas, n_samps = ipe_samps.shape
    shape = (n,) + p_response.shape
    if rso is None:
        rso = np.random
    fall_responses = rso.rand(*shape) < p_response

    # simulate mass? responses
    belief = np.exp(model_theta[1:])
    p_eq = belief[:, kappas == 0.0].sum(axis=1)
    p_heavy = belief[:, kappas > 0.0].sum(axis=1) + 0.5*p_eq
    respond_heavy = rso.rand(n, n_trial) < p_heavy
    mass_responses = np.empty((n, n_trial))
    mass_responses[respond_heavy] = 10.0
    mass_responses[~respond_heavy] = 0.1

    responses = (fall_responses, mass_responses)
    return responses, model_theta
