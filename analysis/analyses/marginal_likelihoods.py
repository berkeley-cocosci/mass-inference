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
__parallel__ = True

import os
import util
import pandas as pd
import numpy as np

from IPython.parallel import Client, require


def as_df(x, index_names):
    df = pd.DataFrame(x)
    if len(index_names) == 1:
        df.index.name = index_names[0]
    else:
        df.index = pd.MultiIndex.from_tuples(df.index)
        df.index.names = index_names
    return df


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


def marginal_likelihood(responses, data, parallel):
    # convert model belief to wide form
    belief = data\
        .set_index(['counterfactual', 'version', 'pid', 'trial', 'hypothesis'])['logp']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute posterior log odds between the two hypotheses, and convert back
    # to long form
    log_odds = pd.melt(
        (belief[1.0] - belief[-1.0]).unstack('trial').reset_index(),
        id_vars=['counterfactual', 'version', 'pid'],
        var_name='trial',
        value_name='log_odds')

    # merge with human responses
    model = pd\
        .merge(log_odds, responses)\
        .set_index(['counterfactual', 'version', 'pid'])\
        .sortlevel()\
        .dropna()
    model['num_mass_trials'] = model\
        .groupby(level=['counterfactual', 'version', 'pid'])\
        .apply(len)
    between_subjs = model\
        .reset_index()\
        .groupby('version')\
        .get_group('I')\
        .groupby(['counterfactual', 'version', 'pid'])\
        .apply(lambda x: x.sort('trial').head(1))\
        .set_index(['counterfactual', 'version', 'pid'])
    between_subjs['num_mass_trials'] = -1

    model = pd\
        .concat([model, between_subjs])\
        .reset_index()\
        .set_index(['counterfactual', 'version', 'num_mass_trials', 'pid', 'trial'])\
        .sortlevel()

    if parallel:
        rc = Client()
        dview = rc[:]
        mapfunc = dview.map_sync
    else:
        mapfunc = map

    # compute marginal likelihoods
    cols = ['counterfactual', 'version', 'num_mass_trials', 'pid']
    result = mapfunc(integrate, list(model.groupby(level=cols)))
    result = as_df(result, cols).reset_index()

    return result


def run(dest, results_path, data_path, parallel):
    # load in raw human mass responses
    human = util.load_human(data_path)['C'][
        ['version', 'kappa0', 'pid', 'trial', 'stimulus', 'mass? response']]
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0

    # load in raw model belief
    store_pth = os.path.abspath(os.path.join(
        results_path, 'model_belief_by_trial.h5'))
    store = pd.HDFStore(store_pth, mode='r')

    sigma, phi = util.get_params()

    results = []
    for key in store.root._v_children:
        print(key)

        if key.startswith("ipe"):
            # look up the name of the key for the parameters that we want (will be
            # something like params_0)
            params = store["/{}/param_ref".format(key)]\
                .reset_index()\
                .set_index(['sigma', 'phi'])['index']\
                .ix[(sigma, phi)]

            pth = "/{}/{}".format(key, params)
        else:
            pth = "/{}/params_0".format(key)

        group = store.root._f_getChild(pth)
        for model in group._v_children:
            data = store["{}/{}".format(pth, model)]
            result = marginal_likelihood(human, data, parallel)
            result['model'] = model
            result['likelihood'] = key
            results.append(result)

    results = pd.concat(results)
    results.to_csv(dest)
    store.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.parallel)
