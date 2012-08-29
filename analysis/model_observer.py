import numpy as np
from scipy.integrate import trapz

import cogphysics.lib.rvs as rvs
import cogphysics.lib.nplib as npl
import cogphysics.lib.circ as circ

import pdb

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

from kde import gen_stability_edges, gen_direction_edges, gen_xy_edges
from kde import stability_nfell_kde, direction_kde, xy_kde

def IME(samps, n_outcomes, predicates):
    """Compute the posterior probability of the outcome given
    kappa using samples from the internal mechanics engine and a
    Jeffrey's Dirichlet prior.

    P(F | S, k) = P(k | F, S) P(F)

    Parameters
    ----------
    samps : array-like (..., n_trials, n_kappas, n_samples)
        samples from the IME
    n_outcomes : int
        number of possible outcomes
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value

    Returns
    -------
    np.ndarray : (..., trials, kappas, outcomes)

    """

    npred = len(predicates)
    ssl = tuple([Ellipsis] + [None]*npred)
    p_outcomes = np.zeros(samps.shape[:-1] + n_outcomes)

    if 'x' in predicates and 'y' in predicates:
        x = predicates.index('x')
        y = predicates.index('y')
        n = (n_outcomes[x], n_outcomes[y])
        sx = np.ma.masked_invalid(samps['x']).filled(0)
        sy = np.ma.masked_invalid(samps['y']).filled(0)
        rsl = list(ssl)
        rsl[x+1] = slice(None)
        rsl[y+1] = slice(None)

        p = xy_kde(sx, sy, n)[rsl]
        p_outcomes += p
        
    for i in xrange(len(predicates)):
        pred = predicates[i]
        n = n_outcomes[i]
        s = samps[pred]
        rsl = list(ssl)
        rsl[i+1] = slice(None)

        if pred == 'stability_nfell':
            p = stability_nfell_kde(s, n)[rsl]
        elif pred == 'direction':
            p = direction_kde(s, n)[rsl]
        elif pred == 'x' or pred == 'y':
            continue
        else:
            raise ValueError, pred

        p_outcomes += p

    p_outcomes = normalize(p_outcomes.reshape(
        samps.shape[:-1] + (-1,)), axis=-1)[1].reshape(
        p_outcomes.shape)

    return p_outcomes

def evaluateFeedback(feedback, p_outcomes, predicates):
    """Evaluate the likelihood of the observed outcomes given the
    probability of each possible outcome.

    P(F_t | S, k)

    Parameters
    ----------
    feedback : array-like (..., n_trials)
        True outcomes of each trial
    p_outcomes : array-like (..., n_trials, n_kappas, n_outcomes)
        Log probability of each observed outcome given kappa
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value

    Returns
    -------
    np.ndarray : (..., n_trials, n_kappas)

    """

    npred = len(predicates)
    n_outcomes = p_outcomes.shape[-npred:]
    ssl = tuple([Ellipsis] + [None]*npred)
    idx = np.ones(feedback.shape + n_outcomes, dtype='bool')

    for i in xrange(len(predicates)):
        pred = predicates[i]
        n = n_outcomes[i]
        rsl = [None]*npred
        rsl[i] = slice(None)
        fb = np.ma.masked_invalid(feedback[pred][ssl])

        if pred == 'stability_nfell':
            edges, binsize = gen_stability_edges(n)
            lo = fb >= edges[:-1][tuple(rsl)]
            hi = fb < edges[1:][tuple(rsl)]
            pidx = (lo & hi).filled(False)
        
        elif pred == 'direction':
            edges, binsize, offset = gen_direction_edges(n)
            nfb = circ.normalize(fb) + offset
            lo = nfb >= edges[:-1][tuple(rsl)]
            hi = nfb < edges[1:][tuple(rsl)]
            pidx = (lo & hi).filled(False)

        elif pred == 'x':
            edges, binsize = gen_xy_edges(n, which='x')
            lo = fb.filled(0) >= edges[:-1][tuple(rsl)]
            hi = fb.filled(0) < edges[1:][tuple(rsl)]
            pidx = lo & hi

        elif pred == 'y':
            edges, binsize = gen_xy_edges(n, which='y')
            lo = fb.filled(0) >= edges[:-1][tuple(rsl)]
            hi = fb.filled(0) < edges[1:][tuple(rsl)]
            pidx = lo & hi

        idx &= pidx

    each = np.expand_dims(idx, axis=-npred-1) * np.exp(p_outcomes)
    shape = p_outcomes.shape[:-npred] + (-1,)
    lh = np.log(np.sum(each.reshape(shape), axis=-1))

    return lh

def learningCurve(feedback, theta0, p_outcomes, predicates):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    prior over mass ratios, and the probability of each possible
    outcome.

    P_t(k) = P(F_t | S, k) P_t-1(k)

    Parameters
    ----------
    feedback : array-like (..., n_trials)
        True outcomes of each trial
    theta0 : array-like (..., 1, n_kappas)
        Log prior probability of each kappa
    p_outcomes : array-like(..., n_trials, n_kappas, n_outcomes)
        Log probability of each possible outcome given kappa
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       likelihoods : np.ndarray (..., n_trials, n_kappas)
       joints : np.ndarray (..., n_trials, n_kappas)
       thetas : np.ndarray (..., n_trials, n_kappas)

    """
    # the likelihood of the true outcome for each trial
    lh = np.concatenate([
        theta0, evaluateFeedback(feedback, p_outcomes, predicates)],
                        axis=-2)
    # unnormalized joint probability of outcomes and mass ratios 
    joint = lh.cumsum(axis=-2)
    # posterior probablity of mass ratios given the outcome 
    thetas = normalize(joint, axis=-1)[1]
    return lh, joint, thetas

######################################################################

def predict(p_outcomes_given_kappa, p_kappas, predicates):
    """Predict the likelihood outcome given the stimulus, P(F_t | S_t)

    Parameters
    ----------
    p_outcomes_given_kappa : array-like (..., n_kappas, n_outcomes)
        log P(F_t | S_t, kappa), obtained from IME
    p_kappas : array-like (..., n_kappas)
        log theta_t = log P_t(kappa)
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value

    """
    npred = len(predicates)
    sl = [Ellipsis] + [None]*npred
    joint = p_outcomes_given_kappa + p_kappas[sl]
    p_outcomes = normalize(joint, axis=-npred-1)[0]
    return p_outcomes

def Loss(n_outcomes, n_responses, predicates, N=1, Cf=10, Cr=6):
    """Loss function for outcomes and responses.

    Parameters
    ----------
    n_outcomes : int
        Number of possible outcomes
    n_responses : int
        Number of possible responses
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value
    N : number (default=1)
        Amount to shift outcome/response values by
    Cf : number (default=10)
        Logistic coefficient for outcomes
    Cr : number (default=6)
        Logistic coefficient for responses

    """

    loss = np.zeros(n_outcomes + (n_responses,))

    for pidx in xrange(len(predicates)):
        n = n_outcomes[pidx]
        pred = predicates[pidx]

        if pred == 'stability_nfell':
            F = np.arange(n)[:, None]
            R = np.arange(n_responses)[None, :]
            sf = ((F + N).astype('f8') / (n-1)) - 0.5
            sr = (R.astype('f8') / (n-1)) - 0.5
            ssf = 1.0 / (1 + np.exp(-Cf * sf))
            ssr = 1.0 / (1 + np.exp(-Cr * sr))
            l = np.sqrt(np.abs(ssf - ssr))

        else:
            l = np.ones((n, n_responses))

        sl = [None]*len(predicates) + [slice(None)]
        sl[pidx] = slice(None)
        loss += l[sl]
            
    return loss

def Risk(p_kappas, p_outcomes_given_kappa, loss, predicates):
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
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value

    """

    # compute marginal probability of outcomes
    p_outcomes = np.exp(predict(
        p_outcomes_given_kappa, p_kappas, predicates))
    # compute expected risk across outcomes
    n_outcomes = p_outcomes.shape[-len(predicates):]
    n_response = loss.shape[-1]
    shape = (p_outcomes.shape[:-len(predicates)] +
             (np.prod(n_outcomes), n_response))
    r = loss * p_outcomes[..., None]
    risk = np.sum(r.reshape(shape), axis=-2)
    return risk

def response(p_kappas, p_outcomes, loss, predicates):
    """Compute optimal responses based on the belief about mass ratio.

    Parameters
    ----------
    p_kappas : array-like (..., n_kappas)
        Probablity of each mass ratio
    p_outcomes : array-like (..., n_kappas, n_outcomes)
        Probability of each outcome given mass ratio
    loss : array-like (..., n_outcomes, n_responses)
        The loss associated with each outcome/response combo
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value

    """
    risk = Risk(p_kappas, p_outcomes, loss, predicates)
    responses = risk.argmin(axis=-1)
    return responses

######################################################################

def ModelObserver(ime_samples, feedback, n_outcomes,
                  predicates, p_outcomes=None, loss=None):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    feedback that they see, and the total number of possible outcomes.

    Parameters
    ----------
    ime_samples : array-like (..., n_trials, n_kappas, n_samples)
        Raw samples from the IME
    feedback : array-like (..., n_trials)
        True outcomes of each trial
    n_outcomes : int
        Number of possible outcomes
    predicates : string ('stability' or 'direction')
        the predicates under which to compute the value
    loss : array-like (..., n_outcomes, n_responses)
        The loss associated with each outcome/response combo

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       likelihoods : np.ndarray (..., n_trials, n_kappas)
       joints : np.ndarray (..., n_trials, n_kappas)
       thetas : np.ndarray (..., n_trials, n_kappas)

    If loss is given, the tuple will have a fourth value, which are
    responses generated by the observer.

    """
    n_kappas = ime_samples.shape[1]
    theta0 = np.log(np.ones(n_kappas, dtype='f8') / n_kappas)[None, :]
    if p_outcomes is None:
        p_outcomes = IME(ime_samples, n_outcomes, predicates)
    lh, joint, thetas = learningCurve(
        feedback, theta0, p_outcomes, predicates)
    if loss is not None:
        resp = response(thetas[:-1], p_outcomes, loss, predicates)
        out = lh, joint, thetas, resp
    else:
        out = lh, joint, thetas
    return out
