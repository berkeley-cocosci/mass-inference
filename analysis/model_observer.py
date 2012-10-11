import numpy as np
from scipy.integrate import trapz

import cogphysics.lib.rvs as rvs
import cogphysics.lib.nplib as npl
import cogphysics.lib.circ as circ

import pdb

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

# from kde import gen_stability_edges, gen_direction_edges, gen_xy_edges
# from kde import stability_nfell_kde, direction_kde, xy_kde

def make_rbf_kernel(alpha, ell):
    def kern(x1, x2):
        term1 = alpha * np.exp((-0.5 * (x1 - x2)**2) / float(ell))
        term2 = 1e-6 * np.abs(x1 - x2)
        k = term1 + term2
        return k
    return kern

def make_gp(x, y, alpha, ell, eps):
    kernel = make_rbf_kernel(alpha=alpha, ell=ell)
    K = kernel(x[:, None], x[None, :]) + (eps * np.eye(x.size))
    Kinv = np.linalg.inv(K)
    def gp(xn):
        Kn = kernel(xn[:, None], x[None, :])
        Knn = kernel(xn[:, None], xn[None, :]) + (eps * np.eye(xn.size))
        y_mean = np.dot(np.dot(Kn, Kinv), y)
        y_cov = Knn - np.dot(np.dot(Kn, Kinv), Kn.T)
        return y_mean, y_cov
    return gp

def make_kde_smoother(x, y, lam):
    def kde_smoother(xn):
        dists = np.abs(xn[:, None] - x[None, :]) / lam
        pdist = np.exp(-0.5 * dists**2) / np.sqrt(2)
        est = np.sum(pdist * y, axis=-1) / np.sum(pdist, axis=-1)
        return est
    return kde_smoother

def IPE(samps, smooth):
    """Compute the posterior probability of the outcome given
    kappa using samples from the internal mechanics engine.

    P(F | S, k) = P(k | F, S) P(F)

    Parameters
    ----------
    samps : array-like (..., n_kappa, n_samples, n_predicates)
        samples from the IPE
    smooth : boolean
        whether to smooth the IPE estimates

    Returns
    -------
    function
        takes values of the form (..., 1, n_conditions, n_predicates) and
        evaluates P(F | S, k)

    """

    # # using center of mass
    # k = rvs.MVGaussian(np.zeros(samps.shape[-1]), np.eye(samps.shape[-1]))
    # n = np.sum(np.any(~np.isnan(samps), axis=-1), axis=-1)[..., None]

    # def f(x):
    #     diff = x[..., None, :] - samps[..., None, :, :]
    #     pdf = ((1-u)*np.ma.masked_invalid(k.PDF(diff / h))) + u
    #     summed = pdf.filled(0).sum(axis=-1)
    #     normed = np.swapaxes(summed / (n*h), -2, -1)
    #     return normed

    if not smooth:
        # using bernoilli fall/not fall
        n = np.sum(~np.isnan(samps), axis=-2)[..., None, :]
        alpha = np.sum(samps, axis=-2)[..., None, :] + 0.5
        beta = np.sum(1-samps, axis=-2)[..., None, :] + 0.5
        pfell = alpha / (alpha + beta)

        def f(x):
            # likelihood of 1 = pfell
            lh1 = x[..., None, :] * pfell[..., None, :, :]
            # likelihood of 0 = 1-pfell
            lh0 = (1-x)[..., None, :] * (1-pfell)[..., None, :, :]
            pdf = np.swapaxes((lh1 + lh0)[..., 0, 0], -2, -1)
            return pdf

    else:
        # using bernoulli fall/not fall plus gaussian process
        # regression
        alpha = np.sum(samps[..., 0], axis=-1) + 0.5
        beta = np.sum(1-samps[..., 0], axis=-1) + 0.5
        pfell_mean = alpha / (alpha + beta)
        pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
        pfell_std = np.sqrt(pfell_var)
        pfell_meanstd = np.mean(pfell_std, axis=-1)

        x = np.arange(0, pfell_mean.size*0.1, 0.1)

        # alph = pfell_meanstd * 10
        # ell = 1. - np.std(pfell_mean)
        # eps = pfell_meanstd ** 2
        # gp = make_gp(x, pfell_mean, alph, ell, eps)
        # pfell = np.clip(gp(x)[0][:, None, None], 0, 1)
        # assert ((pfell >= 0) & (pfell <= 1)).all()

        lam = 0.2
        kde_smoother = make_kde_smoother(x, pfell_mean, lam)
        pfell = kde_smoother(x)[:, None, None]

        def f(x):
            # likelihood of 1 = pfell
            lh1 = x[..., None, :] * pfell[..., None, :, :]
            # likelihood of 0 = 1-pfell
            lh0 = (1-x)[..., None, :] * (1-pfell)[..., None, :, :]
            pdf = np.swapaxes((lh1 + lh0)[..., 0, 0], -2, -1)
            return pdf
        
    return f

def evaluateFeedback(feedback, P_outcomes):
    """Evaluate the likelihood of the observed outcomes given the
    probability of each possible outcome.

    P(F_t | S, k)

    Parameters
    ----------
    feedback : array-like (..., 1, n_conditions, n_predicates)
        True outcomes of each trial (n_conditions is probably n_kappas)
    P_outcomes : function
        should take values of the form (..., n_kappas, n_samples, n_predicates)
        and evaluate the log probability of the outcome given the data
        and kappa P(F | S, k)

    Returns
    -------
    np.ndarray : (..., n_values, n_kappas)

    """

    pf = P_outcomes(feedback)
    lh = np.log(pf)
    return lh

def learningCurve(feedback, ipe_samps, smooth, decay):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    prior over mass ratios, and the probability of each possible
    outcome.

    P_t(k) = P(F_t | S, k) P_t-1(k)

    Parameters
    ----------
    feedback : array-like (..., n_trials)
        True outcomes of each trial
    ipe_samps : array-like (..., n_samples, n_predicates)
        samples from the IPE
    smooth : boolean
        whether to smooth the IPE estimates
    decay : number, 0 <= decay <= 1
        how much to decay the weights on each time step

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       likelihoods : np.ndarray (..., n_trials, n_kappas)
       joints : np.ndarray (..., n_trials, n_kappas)
       thetas : np.ndarray (..., n_trials, n_kappas)

    """

    n_trial = feedback.shape[0]
    n_kappas = ipe_samps.shape[-3]
    lh = None

    for t in xrange(0, n_trial):

        # estimate the density from ipe samples and evaluate the
        # likelihood of the feedback
        f = IPE(ipe_samps[t], smooth=smooth)
        ef = evaluateFeedback(feedback[t], f)

        # allocate arrays
        if lh is None:
            lh = np.empty(ef.shape[:-1] + (n_trial+1, n_kappas))
            lh[..., 0, :] = np.log(1. / n_kappas)
            joint = np.empty(lh.shape)
            joint[..., 0, :] = lh[..., 0, :].copy()

        # the likelihood of the true outcome for each trial
        lh[..., t+1, :] = ef
        # unnormalized joint probability of outcomes and mass ratios
        joint[..., t+1, :] = decay * joint[..., t, :] + lh[..., t+1, :]

    # posterior probablity of mass ratios given the outcome 
    thetas = normalize(joint, axis=-1)[1]

    return lh, joint, thetas

######################################################################

# def predict(p_outcomes_given_kappa, p_kappas, predicates):
#     """Predict the likelihood outcome given the stimulus, P(F_t | S_t)

#     Parameters
#     ----------
#     p_outcomes_given_kappa : array-like (..., n_kappas, n_outcomes)
#         log P(F_t | S_t, kappa), obtained from IPE
#     p_kappas : array-like (..., n_kappas)
#         log theta_t = log P_t(kappa)
#     predicates : string ('stability' or 'direction')
#         the predicates under which to compute the value

#     """
#     npred = len(predicates)
#     sl = [Ellipsis] + [None]*npred
#     joint = p_outcomes_given_kappa + p_kappas[sl]
#     p_outcomes = normalize(joint, axis=-npred-1)[0]
#     return p_outcomes

def Loss(outcomes, n_responses, N=1, Cf=10, Cr=6):
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


    # transform = lambda x: 1 - np.exp(-x)
    # invtransform = lambda y: -np.log(1 - y)
    
    # d0 = np.sqrt(np.sum(outcomes**2, axis=-1)).ravel()
    # d0 = np.sort(transform(d0[~np.isnan(d0)]))
    # arrs = np.array_split(d0, n_responses)
    # edges = invtransform(np.array(
    #     [transform(0)] +
    #     [(arrs[i][-1] + arrs[i+1][0]) / 2.
    #      for i in xrange(n_responses-1)] +
    #     [transform(3)]))

    # centers = (edges[1:] + edges[:-1]) / 2.

    # loss = (centers[:, None] - centers[None, :])**2

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

# def Risk(p_kappas, p_outcomes_given_kappa, loss, predicates):
#     """Compute expected risk for each response given the
#     likelihood of each outcome, the probability of each mass
#     ratio, and the loss associated with each outcome/response
#     combination.

#     Parameters
#     ----------
#     p_kappas : array-like (..., n_kappas)
#         Probablity of each mass ratio
#     p_outcomes_given_kappa : array-like (..., n_kappas, n_outcomes)
#         Probability of each outcome given mass ratio
#     loss : array-like (..., n_outcomes, n_responses)
#         The loss associated with each outcome/response combo
#     predicates : string ('stability' or 'direction')
#         the predicates under which to compute the value

#     """

#     # compute marginal probability of outcomes
#     p_outcomes = np.exp(predict(
#         p_outcomes_given_kappa, p_kappas, predicates))
#     # compute expected risk across outcomes
#     n_outcomes = p_outcomes.shape[-len(predicates):]
#     n_response = loss.shape[-1]
#     shape = (p_outcomes.shape[:-len(predicates)] +
#              (np.prod(n_outcomes), n_response))
#     r = loss * p_outcomes[..., None]
#     risk = np.sum(r.reshape(shape), axis=-2)
#     return risk

# def response(p_kappas, p_outcomes, loss, predicates):
#     """Compute optimal responses based on the belief about mass ratio.

#     Parameters
#     ----------
#     p_kappas : array-like (..., n_kappas)
#         Probablity of each mass ratio
#     p_outcomes : array-like (..., n_kappas, n_outcomes)
#         Probability of each outcome given mass ratio
#     loss : array-like (..., n_outcomes, n_responses)
#         The loss associated with each outcome/response combo
#     predicates : string ('stability' or 'direction')
#         the predicates under which to compute the value

#     """
#     risk = Risk(p_kappas, p_outcomes, loss, predicates)
#     responses = risk.argmin(axis=-1)
#     return responses

######################################################################

def ModelObserver(ipe_samples, feedback, smooth=True, decay=0.99):
    """Computes a learning curve for a model observer, given raw
    samples from their internal 'intuitive mechanics engine', the
    feedback that they see, and the total number of possible outcomes.

    Parameters
    ----------
    ipe_samples : array-like (..., n_trials, n_kappas, n_samples, n_predicates)
        Raw samples from the IPE
    feedback : array-like (..., n_trials, 1, n_conditions, n_predicates)
        True outcomes of each trial (n_conditions is probably n_kappas)
    smooth : boolean
        whether to smooth the IPE estimates
    decay : number, 0 <= decay <= 1 (default=0.99)
        how much to decay the weights on each time step

    Returns
    -------
    tuple : (likelihoods, joints, thetas)
       likelihoods : np.ndarray (..., n_trials, n_kappas)
       joints : np.ndarray (..., n_trials, n_kappas)
       thetas : np.ndarray (..., n_trials, n_kappas)

    If loss is given, the tuple will have a fourth value, which are
    responses generated by the observer.

    """
    n_kappas = ipe_samples.shape[1]
    lh, joint, thetas = learningCurve(feedback, ipe_samples, smooth, decay)
    # if loss is not None:
    #     resp = response(thetas[:-1], p_outcomes, loss, predicates)
    #     out = lh, joint, thetas, resp
    # else:
    out = lh, joint, thetas
    return out
