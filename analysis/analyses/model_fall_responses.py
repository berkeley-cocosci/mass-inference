#!/usr/bin/env python

"""
Computes the average model queries for "will it fall?". The results are saved
into a HDF5 database, with the following structure:

    <query>/params_<n>

where <query> is the name of the query (for example, 'percent_fell') and
params_<n> (e.g. 'params_0') is the particular combination of sigma/phi
parameters. Additionally, there is a <query>/param_ref array that gives a
mapping between the actual parameter values and their identifiers.

Each table in the database has the following columns:

    query (string)
        the model query
    block (string)
        the experiment phase
    kappa0 (float)
        the true log mass ratio
    stimulus (string)
        the name of the stimulus
    lower (float in [0, 1])
        lower bound of the 95% confidence interval
    median (float in [0, 1])
        median value across the bootstrapped means
    upper (float in [0, 1])
        upper bound of the 95% confidence interval
    N (int)
        how many samples the mean was computed over

"""

__depends__ = ["ipe_A", "ipe_B"]
__random__ = True
__parallel__ = True
__ext__ = '.h5'

import util
import pandas as pd
import numpy as np
import os
import model_fall_responses_queries as queries

from IPython.parallel import Client, require


def model_fall_responses(args):
    key, queryname, data = args
    print key

    result = data\
        .groupby(['block', 'stimulus'])\
        .apply(getattr(queries, queryname))\
        .reset_index()\
        .rename(columns=dict(kappa='kappa0'))
    result['query'] = queryname

    return key, result


def run(dest, data_path, parallel, seed):
    np.random.seed(seed)

    # load the raw ipe data
    ipe = util.load_ipe(data_path)

    # open up the store for saving
    store = pd.HDFStore(dest, mode='w')

    # create the ipython parallel client
    if parallel:
        rc = Client()
        lview = rc.load_balanced_view()
        task = require('util', 'model_fall_responses_queries as queries')(model_fall_responses)
    else:
        task = model_fall_responses

    # start the tasks
    all_params = {}
    results = []
    for i, (params, df) in enumerate(ipe.groupby(['sigma', 'phi'])):
        all_params['params_{}'.format(i)] = params

        for query in queries.__all__:
            key = "/{}/params_{}".format(query, i)
            args = [key, query, df]
            if parallel:
                result = lview.apply(task, args)
            else:
                result = task(args)
            results.append(result)

    # save the parameters into the database
    all_params = pd.DataFrame(all_params, index=['sigma', 'phi']).T
    for query in queries.__all__:
        key = "/{}/param_ref".format(query)
        store.append(key, all_params)

    # collect and save results
    while len(results) > 0:
        result = results.pop(0)
        if parallel:
            key, responses = result.get()
            result.display_outputs()
        else:
            key, responses = result

        store.append(key, responses)

    store.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path, args.parallel, args.seed)

