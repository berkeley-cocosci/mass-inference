#!/usr/bin/env python

"""
Creates a new version of each model likelihood for each participant. We need to
do this, because participants complete the trials in a different order -- so 
in particular, for the learning model, we have to put the trials in the right
order for each participant in order to correctly compute how the model learns.

This depends on RESULTS_PATH/model_likelihood.h5, and produces a new HDF5
database, with the same key structure as model_likelihood.h5. For each
table in the database, the columns are:

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
    hypothesis (float)
        hypothesis about the mass ratio
    llh (float)
        log likelihood of the hypothesis

"""

__depends__ = ["trial_order.csv", "model_likelihood.csv"]

import util
import pandas as pd
import os
import numpy as np


def likelihood_by_trial(trials, data):
    llh = data.groupby('kappa0')

    # create an empty dataframe for the results
    results = pd.DataFrame([])

    # iterate through each of the pids
    for (kappa0, _), df in trials.groupby(['kappa0', 'pid']):
        # merge the trial order with the model likelihood
        model = pd.merge(
            llh.get_group(kappa0),
            df.reset_index())

        results = results.append(model, ignore_index=True)

    # make sure hypothesis is of type float, otherwise hdf5 will complain
    results.loc[:, 'hypothesis'] = results['hypothesis'].astype(float)

    return results


def run(dest, results_path):
    # load in trial order
    trial_order = pd.read_csv(os.path.join(
        results_path, 'trial_order.csv'))
    trials = trial_order\
        .groupby('mode')\
        .get_group('experimentC')\
        .drop('mode', axis=1)\
        .set_index('stimulus')\
        .sort_values(by='trial')

    likelihood = pd\
        .read_csv(os.path.join(results_path, 'model_likelihood.csv'))\
        .set_index(['likelihood', 'counterfactual', 'stimulus', 'kappa0', 'hypothesis'])['median']\
        .to_frame('llh')\
        .reset_index()

    results = likelihood_by_trial(trials, likelihood)\
        .set_index(['likelihood', 'counterfactual', 'version', 'pid', 'trial', 'hypothesis'])\
        .sortlevel()

    assert not np.isnan(results['llh']).any()
    assert not np.isinf(results['llh']).any()

    results.to_csv(dest)

if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
