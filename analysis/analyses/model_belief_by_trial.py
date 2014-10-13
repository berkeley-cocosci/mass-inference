#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path
from IPython.parallel import Client, require


@require('numpy', 'pandas', 'sys')
def task(args):
    key, old_store_pth, pth = args
    print key

    sys.path.append(pth)
    from analyses import util
    pd = pandas
    np = numpy

    old_store = pd.HDFStore(old_store_pth, mode='r')
    llh = old_store[key]\
        .set_index(['stimulus'], append=True)['logp']\
        .unstack('hypothesis')

    data = {}

    static = util.normalize(np.asarray(llh), axis=1)[1]
    static_df = llh.copy()
    static_df[:] = static
    static_df = static_df\
        .stack()\
        .reset_index('stimulus')\
        .rename(columns={0: 'logp'})
    static_df['model'] = 'static'
    data['static'] = static_df

    chance = util.normalize(np.zeros_like(llh), axis=1)[1]
    chance_df = llh.copy()
    chance_df[:] = chance
    chance_df = chance_df\
        .stack()\
        .reset_index('stimulus')\
        .rename(columns={0: 'logp'})
    chance_df['model'] = 'chance'
    data['chance'] = chance_df

    llhcum = llh\
        .reset_index('stimulus', drop=True)\
        .stack('hypothesis')\
        .unstack('trial')\
        .cumsum(axis=1)\
        .stack('trial')\
        .unstack('hypothesis')

    learning = util.normalize(np.asarray(llhcum), axis=1)[1]
    learning_df = llh.copy()
    learning_df[:] = learning
    learning_df = learning_df\
        .stack()\
        .reset_index('stimulus')\
        .rename(columns={0: 'logp'})
    learning_df['model'] = 'learning'
    data['learning'] = learning_df

    old_store.close()

    return key, data


def run(results_path, seed):
    np.random.seed(seed)

    old_store_pth = path(results_path).dirname().joinpath(
        'model_likelihood_by_trial.h5').abspath()
    old_store = pd.HDFStore(old_store_pth, mode='r')
    store = pd.HDFStore(results_path, mode='w')

    rc = Client()
    lview = rc.load_balanced_view()
    results = []

    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        result = lview.apply(task, [key, old_store_pth, path.getcwd()])
        results.append(result)

    while len(results) > 0:
        result = results.pop(0)
        if not result.ready():
            result.wait(.1)
        if not result.ready():
            results.append(result)
            continue

        result.display_outputs()
        key, data = result.get()
        for model in data:
            store.append("{}/{}".format(key, model), data[model])

    store.close()
    old_store.close()

if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
