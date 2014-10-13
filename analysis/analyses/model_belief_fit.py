#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path
from scipy.optimize import minimize
from IPython.parallel import Client, require


def log_laplace(x, mu=0, b=1):
    # (1 / 2*b) * np.exp(-np.abs(x - mu) / b)
    c = -np.log(2 * b)
    e = -np.abs(x - mu) / b
    return c + e


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
    df2 = df.dropna()
    y = np.asarray(df2['responses'])
    X = np.asarray(df2['llr'])

    if 'chance' in df.name:
        B = 0
    else:
        B = logistic_regression(X, y, prior_func, verbose)

    mu = 1.0 / (1 + np.exp(-(df['llr'] * B)))

    new_df = df.copy()
    new_df['B'] = B
    new_df['p'] = mu.copy()
    new_df['p correct'] = mu.copy()

    mask = np.asarray(df['kappa0'] < 0)
    new_df.loc[mask, 'p correct'] = 1 - new_df.loc[mask, 'p correct']

    return new_df


@require('numpy', 'pandas', 'sys')
def task(args):
    key, responses, old_store_pth, pth = args
    print key

    sys.path.append(pth)
    from analyses import util
    from analyses import model_belief_fit as mbf
    pd = pandas
    np = numpy

    old_store = pd.HDFStore(old_store_pth, mode='r')
    llh = old_store[key]\
        .set_index(['stimulus', 'model'], append=True)['logp']\
        .unstack('hypothesis')

    # L2 regularization is equivalent to using a laplace prior
    laplace_prior = mbf.make_prior(mbf.log_laplace, mu=1, b=1)

    # compute the log likelihood ratio between the two hypotheses
    llr = llh[-1] - llh[1]

    model = llr\
        .reset_index()\
        .rename(columns={0: 'llr'})\
        .set_index(['version', 'pid', 'trial', 'stimulus'])
    model['responses'] = responses['mass? response']
    model['kappa0'] = responses['kappa0']
    model = model\
        .reset_index()\
        .set_index(['model', 'version', 'pid', 'trial', 'stimulus'])\
        .sortlevel()

    # use L2 logistic regression to fit parameters individually to
    # each participant in version G
    res_G = model\
        .groupby(level='version')\
        .get_group('G')\
        .groupby(level=['model', 'pid'])\
        .apply(mbf.fit_responses, laplace_prior)
    params_G = res_G\
        .reset_index()[['model', 'version', 'kappa0', 'pid', 'B']]\
        .drop_duplicates()\
        .set_index(['model', 'version', 'kappa0', 'pid'])['B']

    # use the parameters fit to participants in version G as a prior
    # over parameters, and then fit parameters to each participant
    empirical_priors = params_G.groupby(level='model').apply(
        lambda x: mbf.make_prior(util.kde, np.asarray(x), 0.1))
    res_ind = model\
        .drop('G', level='version')\
        .groupby(level=['version', 'model', 'pid'])\
        .apply(lambda x: mbf.fit_responses(x, empirical_priors[x.name[1]]))
    params_ind = res_ind\
        .reset_index()[['model', 'version', 'kappa0', 'pid', 'B']]\
        .drop_duplicates()\
        .set_index(['model', 'version', 'kappa0', 'pid'])['B']

    results = res_G.reset_index()
    results['version'] = 'G'
    results = pd\
        .concat([results, res_ind.reset_index()])\
        .dropna()\
        .set_index(['model', 'version', 'kappa0', 'pid', 'trial', 'stimulus'])\
        .sortlevel()\
        .drop('B', axis=1)

    params = pd.concat([params_G, params_ind]).sortlevel()

    return key, results, params


def run(results_path, seed):
    human = util.load_human()

    # load in raw human mass responses
    responses = human['C']\
        .set_index(['version', 'pid', 'trial', 'stimulus'])[['kappa0', 'mass? response']]\
        .sortlevel()
    responses['mass? response'] = (responses['mass? response'] + 1) / 2

    # load in raw model belief
    old_store_pth = path(results_path).dirname().joinpath(
        'model_belief_by_trial.h5').abspath()
    old_store = pd.HDFStore(old_store_pth, mode='r')
    store = pd.HDFStore(results_path, mode='w')

    rc = Client()
    lview = rc.load_balanced_view()
    results = []

    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        result = lview.apply(task, [key, responses, old_store_pth, path.getcwd()])
        results.append(result)

    while len(results) > 0:
        result = results.pop(0)
        if not result.ready():
            results.append(result)
            continue

        result.display_outputs()
        key, belief, params = result.get()
        store.append("{}/belief".format(key), belief)
        store.append("{}/params".format(key), params)

    store.close()
    old_store.close()


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
