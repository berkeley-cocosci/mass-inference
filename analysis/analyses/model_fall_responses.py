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
import pandas
import numpy
import os
import sys

from IPython.parallel import Client, require


def percent_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps / 10.0).apply(util.bootstrap_mean)
    return answer.T


def more_than_half_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 5).apply(util.beta)
    return answer.T


def more_than_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    answer = (samps > 1).apply(util.beta)
    return answer.T


def model_fall_responses(args):
    key, queryname, data, pth = args
    print key

    sys.path.append(pth)
    import model_fall_responses as mfr

    result = data\
        .groupby(['block', 'stimulus'])\
        .apply(getattr(mfr, queryname))\
        .reset_index()\
        .rename(columns=dict(kappa='kappa0'))
    result['query'] = queryname

    return key, result


def run(dest, data_path, parallel, seed):
    numpy.random.seed(seed)

    # load the raw ipe data
    ipe = util.load_ipe(data_path)

    # queries we'll be computing
    queries = ['percent_fell', 'more_than_half_fell', 'more_than_one_fell']

    # open up the store for saving
    store = pandas.HDFStore(dest, mode='w')

    # path to the directory with analysis stuff in it
    pth = os.path.abspath(os.path.dirname(__file__))

    # create the ipython parallel client
    if parallel:
        rc = Client()
        lview = rc.load_balanced_view()
        task = require('numpy', 'pandas', 'sys')(model_fall_responses)
    else:
        task = model_fall_responses

    # start the tasks
    all_params = {}
    results = []
    for i, (params, df) in enumerate(ipe.groupby(['sigma', 'phi'])):
        all_params['params_{}'.format(i)] = params

        for query in queries:
            key = "/{}/params_{}".format(query, i)
            args = [key, query, df, pth]
            if parallel:
                result = lview.apply(task, args)
            else:
                result = task(args)
            results.append(result)

    # save the parameters into the database
    all_params = pandas.DataFrame(all_params, index=['sigma', 'phi']).T
    for query in queries:
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

