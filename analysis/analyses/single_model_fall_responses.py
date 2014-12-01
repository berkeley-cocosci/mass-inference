#!/usr/bin/env python

"""
Pulls out the model's fall responses for the specified sigma and phi from the
database of all fall responses across all parameter values. Depends on the
database RESULTS_PATH/model_fall_responses.h5, and produces a csv file with the
following columns:

    query (string)
        model query
    block (string)
        experiment block (e.g. A, B)
    stimulus (string)
        the stimulus name
    kappa0 (float)
        true log mass ratio
    lower (float)
        lower bound of the 95% confidence interval
    median (float)
        median of the boostrap distribution
    upper (float)
        upper bound of the 95% confidence interval
    N (int)
        how many samples the mean was computed over

"""

__depends__ = ["model_fall_responses.h5"]

import os
import util
import pandas as pd

def run(dest, results_path):
    # open up the database
    store = pd.HDFStore(os.path.abspath(os.path.join(
        results_path, 'model_fall_responses.h5')))

    sigma, phi = util.get_params()

    all_data = pd.DataFrame([])
    for query in store.root._v_children:

        # look up the name of the key for the parameters that we want (will be
        # something like params_0)
        params = store["/{}/param_ref".format(query)]\
            .reset_index()\
            .set_index(['sigma', 'phi'])['index']\
            .ix[(sigma, phi)]

        # load in the data
        data = store["{}/{}".format(query, params)]
        all_data = all_data.append(data)

    all_data = all_data\
        .set_index(['query', 'block', 'stimulus', 'kappa0'])\
        .sortlevel()

    store.close()
    all_data.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
