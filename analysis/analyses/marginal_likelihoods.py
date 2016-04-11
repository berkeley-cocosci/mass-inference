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

from ipyparallel import require


@require('numpy as np', 'pandas as pd')
def integrate(df):
    name, df = df
    df2 = df.dropna()
    y = np.asarray(df2['mass? response'])
    X = np.asarray(df2['log_odds'])

    # range of beta values
    B = np.linspace(-20, 21, 10000)

    # compute the prior
    # (1 / 2*b) * np.exp(-np.abs(x - mu) / b)
    log_prior = -np.log(2) - np.abs(B - 1)

    # compute the likelihood
    p = 1.0 / (1 + np.exp(-(X * B[:, None])))
    log_lh = np.log((y * p) + ((1 - y) * (1 - p))).sum(axis=1)

    # compute the joint and integrate
    joint = np.exp(log_lh + log_prior)
    marginal = np.log(np.trapz(joint, B))

    s = pd.Series([marginal], index=['logp'])
    s.name = name
    return s


def run(dest, results_path, data_path, parallel):
    # load in raw human mass responses
    human = util.load_human(data_path)['C'][
        ['version', 'kappa0', 'pid', 'trial', 'stimulus', 'mass? response']]
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0

    data = pd.read_csv(os.path.join(results_path, 'model_belief_by_trial.csv'))

    # convert model belief to wide form
    cols = ['likelihood', 'counterfactual', 'version', 'model', 'pid']
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
        .set_index(cols)\
        .sortlevel()\
        .dropna()
    model['num_mass_trials'] = model\
        .groupby(level=cols)\
        .apply(len)
    between_subjs = model\
        .reset_index()\
        .groupby('version')\
        .get_group('I')\
        .groupby(cols)\
        .apply(lambda x: x.sort_values(by='trial').head(1))\
        .set_index(cols)
    between_subjs['num_mass_trials'] = -1

    model = pd\
        .concat([model, between_subjs])\
        .reset_index()\
        .set_index(cols + ['num_mass_trials', 'trial'])\
        .sortlevel()

    # compute marginal likelihoods
    mapfunc = util.get_mapfunc(parallel)
    result = mapfunc(integrate, list(model.groupby(level=cols + ['num_mass_trials'])))
    result = util.as_df(result, cols + ['num_mass_trials'])
    result.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.parallel)
