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
__ext__ = '.h5'

import os
import util
import pandas as pd
import numpy as np
import model_belief_by_trial_fit_util as mbf

from IPython.parallel import Client, require, Reference


def model_belief_fit(key, responses, data):
    model_name = key.split("/")[-1]
    print key

    # convert model belief to wide form
    belief = data\
        .set_index(['counterfactual', 'version', 'kappa0', 'pid', 'trial', 'hypothesis'])['logp']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute posterior log odds between the two hypotheses, and convert back
    # to long form
    log_odds = pd.melt(
        (belief[1.0] - belief[-1.0]).unstack('trial').reset_index(),
        id_vars=['counterfactual', 'version', 'kappa0', 'pid'],
        var_name='trial',
        value_name='log_odds')

    # merge with human responses
    model = pd\
        .merge(log_odds, responses)\
        .set_index(['counterfactual', 'version', 'kappa0', 'pid', 'trial'])\
        .sortlevel()\
        .dropna()

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
    raw['B'] = np.nan
    new_belief = pd.concat([fitted, raw])

    return key, new_belief


def run(dest, results_path, data_path, parallel):
    # load in raw human mass responses
    human = util.load_human(data_path)['C'][
        ['version', 'kappa0', 'pid', 'trial', 'stimulus', 'mass? response']]
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0

    # load in raw model belief
    old_store_pth = os.path.abspath(os.path.join(
        results_path, 'model_belief_by_trial.h5'))
    old_store = pd.HDFStore(old_store_pth, mode='r')
    store = pd.HDFStore(dest, mode='w')

    # create the ipython parallel client
    if parallel:
        rc = Client()
        lview = rc.load_balanced_view()
        task = require('numpy as np', 'pandas as pd', 'util', 'model_belief_by_trial_fit_util as mbf')(model_belief_fit)
        rc[:].push(dict(human=human), block=True)
        human = Reference('human')
    else:
        task = model_belief_fit

    # go through each key in the existing database and begin processing it
    results = []
    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        args = [key, human, old_store[key]]
        if parallel:
            result = lview.apply(task, *args)
        else:
            result = task(*args)
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
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.parallel)
