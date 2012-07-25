import numpy as np

import cogphysics.lib.rvs as rvs
import cogphysics.lib.stats as stats
import cogphysics.lib.nplib as npl

import pdb

def IME(samps, n_outcomes, baserate, pseudocount=1):
    """Compute the posterior probability of the outcome given
    kappa using samples from the internal mechanics engine and a
    Jeffrey's Dirichlet prior.

    Parameters
    ----------
    samps : array-like (..., n_trials, n_kappas, n_samples)
        samples from the IME
    n_outcomes : int
        number of possible outcomes
    pseudocount : int (default=1)
        number of pseudocounts

    Returns
    -------
    np.ndarray : (..., trials, kappas, outcomes)

    """

    # # shape for the prior
    # shape = list(samps.shape[:-1]) + [n_outcomes]
    # # Jeffrey's prior
    # prior = rvs.Dirichlet(np.ones(shape) * pseudocount)
    # # conjugate update
    # vecs = samps[..., None] == np.arange(n_outcomes)
    # post = prior.updateMultinomial(vecs, axis=-2)
    # # take the MLE of the posterior's parameters
    # mle = np.log(post.mean.copy())

    counts = np.sum(samps[..., None] == np.arange(n_outcomes), axis=-2) + 1
    #p_k = np.log(counts / np.sum(counts, axis=-2)[..., None, :].astype('f8'))
    p_k = stats.normalize(np.log(counts), axis=-2)[1]
    #baserate = np.sum(counts.reshape((-1, n_outcomes)), axis=0)
    #baserate = baserate / np.sum(baserate).astype('f8')
    mle = stats.normalize(p_k + baserate, axis=-1)[1]
    return mle

def observationDensity(n_outcomes, A=1e10):
    # compute probability that each outcome is actually the outcome
    if not hasattr(A, '__iter__'):
        exp = np.array([A])
    else:
        exp = np.array(A)
    outcomes = np.arange(n_outcomes)
    negdiff = -np.abs(outcomes[:, None] - outcomes[None, :])
    p_obs = stats.normalize(negdiff * np.log(exp[..., None, None]), axis=-2)[1]
    return p_obs

def evaluateObserved(p_outcomes, A=1e10):
    """Evaluate the probability of an observed outcome given the
    probability of each outcome according to the IME.

    Parameters
    ----------
    p_outcomes : array-like (..., n_trials, n_kappas, n_outcomes)
        Log probability of each observed outcome given kappa
    A : number
        P(F' | F) parameter

    Returns
    -------
    np.ndarray : (..., n_trials, n_kappas)

    """
    # total number of possible outcomes
    n_outcomes = p_outcomes.shape[-1]
    p_obs = observationDensity(n_outcomes, A=A)
    # compute probability of each possible observed outcome (different
    # from actual outcome)
    joint = p_obs[..., None, :, :] + p_outcomes[..., None, :]
    p_obs_outcomes = stats.normalize(joint, axis=-1)[0]
    return p_obs_outcomes

def evaluateTruth(truth, p_obs_outcomes):
    """Evaluate the likelihood of the observed outcomes given the
    probability of each possible outcome.

    Parameters
    ----------
    truth : array-like (..., n_trials)
        True outcomes of each trial
    p_obs_outcomes : array-like (..., n_trials, n_kappas, n_outcomes)
        Log probability of each observed outcome given kappa

    Returns
    -------
    np.ndarray : (..., n_trials, n_kappas)

    """
    # total number of possible outcomes
    n_outcomes = p_obs_outcomes.shape[-1]
    # multinomial random variable with parameters equal to P(kappa)
    mn = rvs.Multinomial(1, np.exp(p_obs_outcomes))
    # turn truth values into multinomial vectors
    vecs = (truth[..., None] == np.arange(n_outcomes)).astype('int')
    # compute log likelihood of truth
    lh = stats.normalize(mn.logPMF(vecs[..., None, :]), axis=-1)[1]
    return lh

def learningCurve(truth, theta0, p_obs_outcomes):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    prior over mass ratios, and the probability of each possible
    outcome.

    Parameters
    ----------
    truth : array-like (..., n_trials)
        True outcomes of each trial
    theta0 : array-like (..., 1, n_kappas)
        Log prior probability of each kappa
    p_obs_outcomes : array-like(..., n_trials, n_kappas, n_outcomes)
        Log probability of each possible outcome given kappa

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       likelihoods : np.ndarray (..., n_trials, n_kappas)
       joints : np.ndarray (..., n_trials, n_kappas)
       thetas : np.ndarray (..., n_trials, n_kappas)

    """
    # the likelihood of the true outcome for each trial
    lh = np.concatenate([
        theta0, evaluateTruth(truth, p_obs_outcomes)],
                        axis=-2)
    # unnormalized joint probability of outcomes and mass ratios 
    joint = lh.cumsum(axis=-2)
    # posterior probablity of mass ratios given the outcome 
    thetas = stats.normalize(joint, axis=-1)[1]
    return lh, joint, thetas

######################################################################

def predict(p_outcomes_given_kappa, p_kappas):
    """Predict the likelihood outcome given the stimulus, P(F_t | S_t)

    Parameters
    ----------
    p_outcomes_given_kappa : array-like (..., n_kappas, n_outcomes)
        log P(F_t | S_t, kappa), obtained from IME
    p_kappas : array-like (..., n_kappas)
        log theta_t = log P_t(kappa)
    axis : int (default=0)
        axis along which to normalize, i.e. kappa dimension

    """
    joint = p_outcomes_given_kappa + p_kappas[..., None]
    p_outcomes = stats.normalize(joint, axis=-2)[0]
    return p_outcomes

def Loss(n_outcomes, n_responses, N=1, Cf=10, Cr=6):
    """Loss function for outcomes and responses.

    Parameters
    ----------
    n_outcomes : int
        Number of possible outcomes
    n_responses : int
        Number of possible responses
    N : number (default=1)
        Amount to shift outcome/response values by
    Cf : number (default=10)
        Logistic coefficient for outcomes
    Cr : number (default=6)
        Logistic coefficient for responses

    """

    F = np.arange(n_outcomes)[:, None]
    R = np.arange(n_responses)[None, :]
    sf = ((F + N).astype('f8') / (n_outcomes-1)) - 0.5
    sr = (R.astype('f8') / (n_responses-1)) - 0.5
    ssf = 1.0 / (1 + np.exp(-Cf * sf))
    ssr = 1.0 / (1 + np.exp(-Cr * sr))
    loss = np.sqrt(np.abs(ssf - ssr))
    return loss

def Risk(p_kappas, p_outcomes_given_kappa, loss):
    """Compute expected risk for each response given the
    likelihood of each outcome, the probability of each mass
    ratio, and the loss associated with each outcome/response
    combination.

    Parameters
    ----------
    p_kappas : array-like (..., n_kappas)
        Probablity of each mass ratio
    p_outcomes_given_kappa : array-like (..., n_kappas, n_outcomes)
        Probability of each outcome given mass ratio
    loss : array-like (..., n_outcomes, n_responses)
        The loss associated with each outcome/response combo
    kaxis : int (default=0)
        kappa dimension
    faxis : int (default=0)
        outcome dimension

    """

    # compute marginal probability of outcomes
    p_outcomes = np.exp(predict(p_outcomes_given_kappa, p_kappas))
    # compute expected risk across outcomes
    risk = (np.ma.masked_invalid(loss) * p_outcomes[..., None]).sum(axis=-2)
    return np.ma.filled(risk, fill_value=np.inf)

def response(p_kappas, p_outcomes, loss):
    """Compute optimal responses based on the belief about mass ratio.

    Parameters
    ----------
    p_kappas : array-like (..., n_kappas)
        Probablity of each mass ratio
    p_outcomes : array-like (..., n_kappas, n_outcomes)
        Probability of each outcome given mass ratio
    loss : array-like (..., n_outcomes, n_responses)
        The loss associated with each outcome/response combo

    """
    risk = Risk(p_kappas, p_outcomes, loss)
    responses = risk.argmin(axis=-1)
    return responses

######################################################################

def ModelObserver(ime_samples, truth, baserate, n_outcomes, A, loss=None):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    feedback that they see, and the total number of possible outcomes.

    Parameters
    ----------
    ime_samples : array-like (..., n_trials, n_kappas, n_samples)
        Raw samples from the IME
    truth : array-like (..., n_trials)
        True outcomes of each trial
    n_outcomes : int
        Number of possible outcomes

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       likelihoods : np.ndarray (..., n_trials, n_kappas)
       joints : np.ndarray (..., n_trials, n_kappas)
       thetas : np.ndarray (..., n_trials, n_kappas)

    """
    n_kappas = ime_samples.shape[1]
    theta0 = np.log(np.ones(n_kappas, dtype='f8') / n_kappas)[None, :]
    p_outcomes = IME(ime_samples, n_outcomes, baserate)
    p_obs_outcomes = evaluateObserved(p_outcomes, A)
    lh, joint, thetas = learningCurve(
        truth, theta0, p_obs_outcomes)
    if loss is not None:
        resp = response(thetas[:-1], p_obs_outcomes, loss)
        out = lh, joint, thetas, resp
    else:
        out = lh, joint, thetas
    return out
