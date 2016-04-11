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

__depends__ = ["human", "model_belief_by_trial.csv"]
__parallel__ = True

import os
import util
import pandas as pd
import numpy as np
import scipy.optimize

from ipyparallel import require


@require('numpy as np', 'pandas as pd', 'scipy.optimize')
def fit_responses(df):
    """Fits participant responses using a logistic regression. The given data
    frame should have, at least, columns for 'mass? response' and 'log_odds'. A
    new dataframe will be returned, minus columns for 'mass? response' and
    'log_odds', but with columns 'B' (the fitted parameter), 'p' (the fitted
    belief for r=10), and 'p correct' (the fitted probability of answering
    correctly).

    """
    (likelihood, counterfactual, version, model, kappa0, pid), df = df

    df2 = df.dropna()
    y = np.asarray(df2['mass? response'])
    X = np.asarray(df2['log_odds'])

    def log_posterior(B):
        # compute the prior
        # (1 / 2*b) * np.exp(-np.abs(x - mu) / b)
        log_prior = -np.log(2) - np.abs(B - 1)

        # compute the likelihood
        p = 1.0 / (1 + np.exp(-(X * B)))
        log_lh = np.log((y * p) + ((1 - y) * (1 - p))).sum()

        # compute the posterior
        log_posterior = log_lh + log_prior

        return -log_posterior

    B = float(scipy.optimize.minimize_scalar(log_posterior)['x'])
    f = np.asarray(df['log_odds']) * B
    f_raw = np.asarray(df['log_odds'])
    mu = 1.0 / (1 + np.exp(-f))
    mu_raw = 1.0 / (1 + np.exp(-f_raw))

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


def run(dest, results_path, data_path, parallel):
    # load in raw human mass responses
    human = util.load_human(data_path)['C'][
        ['version', 'kappa0', 'pid', 'trial', 'stimulus', 'mass? response']]
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0

    data = pd.read_csv(os.path.join(results_path, 'model_belief_by_trial.csv'))

    # convert model belief to wide form
    cols = ['likelihood', 'counterfactual', 'version', 'model', 'kappa0', 'pid']
    belief = data\
        .set_index(cols + ['trial', 'hypothesis'])['logp']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute posterior log odds between the two hypotheses, and convert back
    # to long form
    log_odds = pd.melt(
        (belief[1.0] - belief[-1.0]).unstack('trial').reset_index(),
        id_vars=cols,
        var_name='trial',
        value_name='log_odds')

    # merge with human responses
    model = pd\
        .merge(log_odds, human)\
        .set_index(cols + ['trial'])\
        .sortlevel()\
        .dropna()

    # use L1 logistic regression to fit parameters individually to
    # each participant
    mapfunc = util.get_mapfunc(parallel)
    result = mapfunc(fit_responses, list(model.groupby(level=cols)))
    result = pd.concat(result).reset_index()

    # separate out the raw belief from the fitted belief
    fitted = result.drop(['p raw', 'p correct raw'], axis=1)
    fitted['fitted'] = True
    raw = result\
        .drop(['p', 'p correct'], axis=1)\
        .rename(columns={'p raw': 'p', 'p correct raw': 'p correct'})
    raw['fitted'] = False
    raw['B'] = np.nan

    new_belief = pd\
        .concat([fitted, raw])\
        .set_index(cols)\
        .sortlevel()

    assert not np.isnan(new_belief['p']).any()
    assert not np.isnan(new_belief['p correct']).any()
    assert not np.isinf(new_belief['p']).any()
    assert not np.isinf(new_belief['p correct']).any()

    new_belief.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.parallel)
