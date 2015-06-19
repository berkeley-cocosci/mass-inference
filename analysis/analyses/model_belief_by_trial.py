#!/usr/bin/env python

"""
Computes the model belief for each participant for the following models:

    * static
    * learning

This depends on RESULTS_PATH/model_likelihood_by_trial.h5, and will produce a
new database similar to that one. For each key in the likelihood database, this
will have keys named <key>/static, <key>/learning, and <key>/static. For each
one of these tables, the columns are:

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
    logp (float)
        log posterior probability of the hypothesis

"""

__depends__ = ["model_likelihood_by_trial.csv"]

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path):
    data = pd.read_csv(os.path.join(results_path, 'model_likelihood_by_trial.csv'))

    # compute the belief
    cols = ['likelihood', 'counterfactual', 'version', 'pid']
    llh = data\
        .set_index(cols + ['hypothesis', 'trial'])['llh']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute the belief for each model
    models = {
        'static': llh.copy(),
        'learning': llh.groupby(level=cols).apply(np.cumsum),
    }

    results = pd.DataFrame([])
    for model_name, model in models.items():
        # normalize the probabilities so they sum to one
        model[:] = util.normalize(
            np.asarray(model), axis=1)[1]

        # convert to long form
        model = pd.melt(
            model.reset_index(),
            id_vars=cols + ['trial'],
            var_name='hypothesis',
            value_name='logp')

        # merge with the existing data
        model = pd.merge(data, model).drop('llh', axis=1)
        model['model'] = model_name
        results = results.append(model)

    results = results\
        .set_index(cols + ['model', 'trial', 'hypothesis'])\
        .sortlevel()

    assert not np.isnan(results['logp']).any()
    assert not np.isinf(results['logp']).any()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
