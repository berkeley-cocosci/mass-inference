#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path
from scipy.optimize import minimize

def log_laplace(x, mu=0, b=1):
    # (1 / 2*b) * np.exp(-np.abs(x - mu) / b)
    c = -np.log(2 * b)
    e = -np.abs(x - mu) / b
    return c + e

def kde(x, params, sigma):
    p = -0.5 * np.log(2 * np.pi * sigma**2) - (x - params) ** 2 / (2 * sigma ** 2)
    logp = np.log(np.mean(np.exp(p), axis=-1))
    return logp

def make_prior(f, *args, **kwargs):
    def prior(x):
        return f(x, *args, **kwargs)
    return prior

def make_posterior(X, y, prior_func, verbose=False):
    def log_posterior(B):
        p = 1.0 / (1 + np.exp(-(X * B)))
        log_lh = np.log((y * p) + ((1 - y) * (1 - p))).sum()
        log_prior = prior_func(B)
        log_posterior = log_lh + log_prior
        if verbose:
            print B, -log_posterior
        return -log_posterior
    return log_posterior

def logistic_regression(X, y, prior_func, verbose=False):
    log_posterior = make_posterior(X, y, prior_func, verbose)
    res = [minimize(fun=log_posterior, x0=x0) for x0 in [-1.0, 0.0, 1.0, 2.0]]
    res.sort(cmp=lambda x, y: cmp(x['fun'], y['fun']))
    best = res[0]
    if verbose:
        print "best:", best['x'], best['fun']
    return float(best['x'])

def fit_responses(df, prior_func, verbose=False):
    y = np.asarray(df['responses'])
    X = np.asarray(df['llr'])
    
    if 'chance' in df.name:
        B = 0
    else:
        B = logistic_regression(X, y, prior_func, verbose)
        
    f = X * B
    mu = 1.0 / (1 + np.exp(-f))
        
    new_df = df.copy()
    new_df['f'] = f
    new_df['B'] = B
    new_df['p'] = mu.copy()
    new_df['p correct'] = mu.copy()

    mask = np.asarray(df['kappa0'] < 0)
    new_df.loc[mask, 'p correct'] = 1 - new_df.loc[mask, 'p correct']

    return new_df

def load(results_path):
    human = util.load_human()

    # load in raw human mass responses
    responses = human['C']\
        .set_index(['version', 'pid', 'trial', 'stimulus'])['mass? response']\
        .dropna()\
        .sortlevel()
    responses = (responses + 1) / 2

    # load in human mass accuracy
    correct = human['C']\
        .set_index(['version', 'pid', 'trial', 'stimulus'])['mass? correct']\
        .dropna()\
        .sortlevel()

    # load in the correct hypothesis
    kappa0 = human['C']\
        .set_index(['version', 'pid', 'trial', 'stimulus'])['kappa0']\
        .ix[correct.index]

    # load in raw model belief
    belief = pd.read_csv("results/model_belief_by_trial.csv")

    # TODO: need to run this for all priors, not just empirical
    query = util.get_query()
    llh = belief\
        .groupby(['likelihood', 'query'])\
        .get_group(('empirical', query))\
        .drop(['sigma', 'phi'], axis=1)\
        .set_index(['version', 'model', 'pid', 'trial', 'stimulus', 'hypothesis'])['logp']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute the log likelihood ratio between the two hypotheses
    llr = (llh[1] - llh[-1])\
        .unstack('model')\
        .ix[responses.index]

    # build up a dataframe with all the relevant information
    model = llr.copy()
    model['responses'] = responses
    model['correct'] = correct
    model['kappa0'] = kappa0
    model = model\
        .set_index(['responses', 'correct', 'kappa0'], append=True)\
        .stack()\
        .reset_index(['responses', 'correct', 'kappa0'])\
        .rename(columns={0: 'llr'})
    model.index.names = ['version', 'pid', 'trial', 'stimulus', 'model']
    model = model\
        .reorder_levels(['version', 'model', 'pid', 'trial', 'stimulus'])\
        .sortlevel()

    return model

def run(results_path, seed):
    np.random.seed(seed)
    model = load(results_path)

    # L2 regularization is equivalent to using a laplace prior
    laplace_prior = make_prior(log_laplace, mu=1, b=1)

    # # use L2 logistic regression to fit a global parameter to all
    # # participants in aggregate
    # res_all = model\
    #     .groupby(level=['version', 'model'])\
    #     .apply(fit_responses, laplace_prior)
    # params_all = res_all\
    #     .reset_index()[['version', 'model', 'B']]\
    #     .drop_duplicates()\
    #     .set_index(['version', 'model'])

    # use L2 logistic regression to fit parameters individually to
    # each participant in version G
    res_G = model\
        .groupby(level='version')\
        .get_group('G')\
        .groupby(level=['model', 'pid'])\
        .apply(fit_responses, laplace_prior)
    params_G = res_G\
        .reset_index()[['model', 'pid', 'B']]\
        .drop_duplicates()\
        .set_index(['model', 'pid'])['B']

    # use the parameters fit to participants in version G as a prior
    # over parameters, and then fit parameters to each participant
    empirical_priors = params_G.groupby(level='model').apply(
        lambda x: make_prior(kde, np.asarray(x), 0.1))
    res_ind = model\
        .groupby(level=['version', 'model', 'pid'])\
        .apply(lambda x: fit_responses(x, empirical_priors[x.name[1]]))
    params_ind = res_ind\
        .reset_index()[['version', 'model', 'pid', 'B']]\
        .drop_duplicates()\
        .set_index(['version', 'model', 'pid'])

    results = res_G.reset_index()
    results['version'] = 'G'
    results = pd.concat([results, res_ind.drop('G', level='version').reset_index()])
    results['likelihood'] = 'empirical'
    results['query'] = util.get_query()
    results = results\
        .set_index(
            ['model', 'likelihood', 'version', 'trial', 'query',
             'kappa0', 'pid', 'stimulus'])[['p', 'p correct', 'B']]\
        .sortlevel()

    results.to_csv(results_path)

if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
