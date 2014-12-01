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
__ext__ = '.h5'

import util
import pandas as pd
import numpy as np


def percent_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    return (samps / 10.0).apply(util.bootstrap_mean)


def more_than_half_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    return (samps > 5).apply(util.beta)


def more_than_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    return (samps > 1).apply(util.beta)


def save(data, pth, store):
    params = ['sigma', 'phi']
    all_params = {}
    for i, (p, df) in enumerate(data.groupby(level=params)):
        key = '{}/params_{}'.format(pth, i)
        print key
        df2 = df.reset_index(params, drop=True)
        store.append(key, df2)
        all_params['params_{}'.format(i)] = p
    all_params = pd.DataFrame(all_params, index=params).T
    store.append('{}/param_ref'.format(pth), all_params)


def compute_query(data, query, store):
    results = []
    for block in ['A', 'B']:
        result = data[block]\
            .groupby(level=['sigma', 'phi', 'stimulus'])\
            .apply(query)\
            .unstack()\
            .stack('kappa')\
            .reset_index()\
            .rename(columns={
                'kappa': 'kappa0'
            })
        result['query'] = query.__name__
        result['block'] = block
        results.append(result)

    results = pd.concat(results)
    save(results.set_index(['sigma', 'phi']), query.__name__, store)


def run(dest, data_path, seed):
    np.random.seed(seed)
    ipe = util.load_ipe(data_path)

    store = pd.HDFStore(dest, mode='w')
    compute_query(ipe, percent_fell, store)
    compute_query(ipe, more_than_half_fell, store)
    compute_query(ipe, more_than_one_fell, store)
    store.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path, args.seed)

