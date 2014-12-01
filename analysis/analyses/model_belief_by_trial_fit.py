#!/usr/bin/env python

"""
Fits the model belief to human responses using a logistic regression. For each
participant, we compute:

    p = 1 / (1 + exp(-BX))

where B is the learning rate parameter, and X is the posterior log odds, i.e.

    X = log(Pr_t(r | F_t, S_t) / Pr_t(r' | F_t, S_t))

We then compute the best parameter B according to:

    argmax_{B} [(p * y) + (1 - p)*(1 - y)] * Pr(B)

where y is the participant's judgment, and Pr(B) is a laplace prior (equivalent
to L1 regularization). The p calculated from the resulting value of B is our new
fitted belief.

This relies on the RESULTS_PATH/model_belief_by_trial.h5 database, and produces
a similar database with the same keys. Each table in the database has the
following columns (it includes both fitted beliefs, and raw beliefs):

    counterfactual (bool)
        whether the counterfactual likelihood was used
    version (string)
        the experiment version
    kappa0 (float)
        true log mass ratio
    pid (string)
        unique participant id
    stimulus (string)
        stimulus name
    trial (int)
        trial number
    p (float)
        posterior probability of the hypothesis that r=10
    p correct (float)
        posterior probability of the correct hypothesis
    B (float)
        fitted parameter for the logistic regression
    fitted (bool)
        whether this version of the model was fit or not

"""

__depends__ = ["human", "model_belief_by_trial.h5"]

import os
import sys
import util
import pandas
import numpy

from scipy.optimize import minimize
from IPython.parallel import Client, require


def log_laplace(x, mu=0, b=1):
    """Compute p(x | mu, b) according to a laplace distribution"""
    # (1 / 2*b) * numpy.exp(-numpy.abs(x - mu) / b)
    c = -numpy.log(2 * b)
    e = -numpy.abs(x - mu) / b
    return c + e


def make_posterior(X, y, verbose=False):
    """Returns a function that takes an argument for the hypothesis, and that
    then computes the posterior probability of that hypothesis given X and y

    """
    def f(B):
        # compute the prior
        log_prior = log_laplace(B, mu=1, b=1)

        # compute the likelihood
        p = 1.0 / (1 + numpy.exp(-(X * B)))
        log_lh = numpy.log((y * p) + ((1 - y) * (1 - p))).sum()

        # compute the posterior
        log_posterior = log_lh + log_prior

        if verbose:
            print B, -log_posterior
        
        return -log_posterior

    return f


def logistic_regression(X, y, verbose=False):
    """Performs a logistic regression with one coefficient for predictors X and
    output y.

    """
    log_posterior = make_posterior(X, y, verbose)
    res = [minimize(fun=log_posterior, x0=x0) for x0 in [-1.0, 0.0, 1.0, 2.0]]
    res.sort(cmp=lambda x, y: cmp(x['fun'], y['fun']))
    best = res[0]
    if verbose:
        print "best:", best['x'], best['fun']
    return float(best['x'])


def fit_responses(df, model_name, verbose=False):
    """Fits participant responses using a logistic regression. The given data
    frame should have, at least, columns for 'mass? response' and 'log_odds'. A
    new dataframe will be returned, minus columns for 'mass? response' and
    'log_odds', but with columns 'B' (the fitted parameter), 'p' (the fitted
    belief for r=10), and 'p correct' (the fitted probability of answering
    correctly).

    """
    counterfactual, version, kappa0, pid = df.name

    df2 = df.dropna()
    y = numpy.asarray(df2['mass? response'])
    X = numpy.asarray(df2['log_odds'])

    if model_name == 'chance':
        B = 0
    else:
        B = logistic_regression(X, y, verbose)

    f = numpy.asarray(df['log_odds']) * B
    f_raw = numpy.asarray(df['log_odds'])
    mu = 1.0 / (1 + numpy.exp(-f))
    mu_raw = 1.0 / (1 + numpy.exp(-f_raw))

    new_df = df.copy().drop(['mass? response', 'log_odds'], axis=1)
    new_df['B'] = B
    new_df['p'] = mu
    new_df['p raw'] = mu_raw

    if kappa0 < 0:
        new_df['p correct'] = 1 - mu
        new_df['p correct raw'] = 1 - mu_raw
    else:
        new_df['p correct'] = mu
        new_df['p correct raw'] = mu_raw

    return new_df


def model_belief_fit(args):
    key, responses, old_store_pth, pth = args
    model_name = key.split("/")[-1]
    print key

    # import util and model_belief_by_trial_fit
    sys.path.append(pth)
    import util
    import model_belief_by_trial_fit as mbf

    old_store = pandas.HDFStore(old_store_pth, mode='r')
    data = old_store[key]
    old_store.close()

    # convert model belief to wide form
    belief = data\
        .set_index(['counterfactual', 'version', 'kappa0', 'pid', 'trial', 'hypothesis'])['logp']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute posterior log odds between the two hypotheses, and convert back
    # to long form
    log_odds = pandas.melt(
        (belief[1.0] - belief[-1.0]).unstack('trial').reset_index(),
        id_vars=['counterfactual', 'version', 'kappa0', 'pid'],
        var_name='trial',
        value_name='log_odds')

    # merge with human responses
    model = pandas\
        .merge(log_odds, responses)\
        .set_index(['counterfactual', 'version', 'kappa0', 'pid', 'trial'])\
        .sortlevel()

    # use L1 logistic regression to fit parameters individually to
    # each participant
    result = model\
        .groupby(level=['counterfactual', 'version', 'kappa0', 'pid'])\
        .apply(mbf.fit_responses, model_name)\
        .reset_index()

    # separate out the raw belief from the fitted belief
    fitted = result.drop(['p raw', 'p correct raw'], axis=1)
    fitted['fitted'] = True
    raw = result\
        .drop(['p', 'p correct'], axis=1)\
        .rename(columns={'p raw': 'p', 'p correct raw': 'p correct'})
    raw['fitted'] = False
    raw['B'] = numpy.nan
    new_belief = pandas.concat([fitted, raw])

    return key, new_belief


def run(dest, results_path, data_path, parallel):
    # load in raw human mass responses
    human = util.load_human(data_path)['C'][
        ['version', 'kappa0', 'pid', 'trial', 'stimulus', 'mass? response']]
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0

    # load in raw model belief
    old_store_pth = os.path.abspath(os.path.join(
        results_path, 'model_belief_by_trial.h5'))
    old_store = pandas.HDFStore(old_store_pth, mode='r')
    store = pandas.HDFStore(dest, mode='w')

    # path to the directory with analysis stuff in it
    pth = os.path.abspath(os.path.dirname(__file__))

    # create the ipython parallel client
    if parallel:
        rc = Client()
        lview = rc.load_balanced_view()
        task = require('numpy', 'pandas', 'sys')(model_belief_fit)
    else:
        task = model_belief_fit

    # go through each key in the existing database and begin processing it
    results = []
    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        args = [key, human, old_store_pth, pth]
        if parallel:
            result = lview.apply(task, args)
        else:
            result = task(args)
        results.append(result)

    # collect and save results
    while len(results) > 0:
        result = results.pop(0)
        if parallel:
            key, belief = result.get()
            result.display_outputs()
        else:
            key, belief = result

        store.append(key, belief)

    store.close()
    old_store.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals(), parallel=True, ext=".h5")
    args = parser.parse_args()
    run(args.dest, args.results_path, args.data_path, args.parallel)
