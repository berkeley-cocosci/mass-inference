#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def make_model_df(arr, trials, hyps, kappa0, pid, version):
    new_df = pd.DataFrame(arr, index=trials['stimulus'], columns=hyps)
    new_df.index.name = 'stimulus'
    new_df.columns.name = 'hypothesis'
    new_df['trial'] = np.asarray(trials['trial'])
    new_df = new_df.set_index('trial', append=True).stack()
    new_df.name = 'logp'
    new_df = new_df.reset_index()
    new_df['kappa0'] = kappa0
    new_df['pid'] = pid
    new_df['version'] = version
    new_df = new_df.set_index([
        'version', 'kappa0', 'pid', 'trial', 'hypothesis'])
    return new_df


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_human()

    old_store = pd.HDFStore(path(results_path).dirname().joinpath(
        'model_likelihood.h5'))
    store = pd.HDFStore(results_path, mode='w')

    pth = path(results_path).dirname().joinpath("trial_order.csv")
    trials = pd.read_csv(pth)\
               .set_index(['mode', 'trial'])\
               .ix['experimentC']

    human = data['C']
    participants = human[['version', 'kappa0', 'pid']]\
        .drop_duplicates()\
        .sort(['version', 'kappa0', 'pid'])\
        .set_index(['version', 'kappa0'])['pid']

    orders = {}
    for pid in participants:
        order = trials[pid].dropna()
        order.name = 'stimulus'
        order = order.reset_index()
        i = np.argsort(order.sort('stimulus')['trial'])
        orders[pid] = (order, i)

    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        llh = old_store[key]\
            .reset_index('kappa0')
        llh['kappa0'] = llh['kappa0'].apply(str)
        llh = llh\
            .set_index('kappa0', append=True)\
            .reorder_levels(['kappa0', 'stimulus'])
        hyps = llh.columns

        for (version, kappa0), pids in participants.groupby(level=['version', 'kappa0']):
            print key, version, kappa0

            df = np.asarray(llh.ix[str(kappa0)])

            for pid in pids:
                order, i = orders[pid]
                dfi = df[i]

                model = make_model_df(dfi, order, hyps, kappa0, pid, version)
                store.append(key, model)

    store.close()
    old_store.close()

if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
