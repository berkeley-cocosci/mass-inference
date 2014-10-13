#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np


def run(results_path, seed):
    np.random.seed(seed)

    old_store = pd.HDFStore('results/model_likelihood_by_trial.h5')
    store = pd.HDFStore(results_path, mode='w')

    for key in old_store.keys():
        print key
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        llh = old_store[key]\
            .set_index(['stimulus'], append=True)['logp']\
            .unstack('hypothesis')

        static = util.normalize(np.asarray(llh), axis=1)[1]
        static_df = llh.copy()
        static_df[:] = static
        static_df = static_df\
            .stack()\
            .reset_index('stimulus')\
            .rename(columns={0: 'logp'})
        static_df['model'] = 'static'
        store.append("{}/static".format(key), static_df)

        chance = util.normalize(np.zeros_like(llh), axis=1)[1]
        chance_df = llh.copy()
        chance_df[:] = chance
        chance_df = chance_df\
            .stack()\
            .reset_index('stimulus')\
            .rename(columns={0: 'logp'})
        chance_df['model'] = 'chance'
        store.append("{}/chance".format(key), chance_df)

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
        store.append("{}/learning".format(key), learning_df)

    store.close()

if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
