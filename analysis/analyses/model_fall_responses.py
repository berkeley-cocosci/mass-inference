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

__depends__ = ["model_fall_responses_raw.h5"]
__random__ = True
__parallel__ = True
__ext__ = '.h5'

import util
import pandas as pd
import numpy as np
import os

from IPython.parallel import Client, require


def model_fall_responses(key, ipe):
    print key
    samps = ipe.set_index(['query', 'block', 'kappa0', 'stimulus', 'sample'])['response'].unstack('sample')
    if ((samps == 0) | (samps == 1)).all():
        result = samps.apply(util.beta, axis=1)
    else:
        result = samps.apply(util.bootstrap_mean, axis=1)
    result['mean'] = samps.mean(axis=1)
    result['stddev'] = samps.std(axis=1)
    result = result.reset_index()
    return key, result


def run(dest, results_path, parallel, seed):
    np.random.seed(seed)

    # load the responses
    old_store_pth = os.path.abspath(os.path.join(
        results_path, 'model_fall_responses_raw.h5'))
    old_store = pd.HDFStore(old_store_pth, mode='r')

    # open up the store for saving
    store = pd.HDFStore(dest, mode='w')

    # create the ipython parallel client
    if parallel:
        rc = Client()
        lview = rc.load_balanced_view()
        task = require('util')(model_fall_responses)
    else:
        task = model_fall_responses

    # start the tasks
    results = []
    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        args = [key, old_store[key]]
        if parallel:
            result = lview.apply(task, *args)
        else:
            result = task(*args)
        results.append(result)

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
    run(args.to, args.results_path, args.parallel, args.seed)

