#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_belief.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    results = []

    trials = pd.read_csv(results_path.joinpath('trial_order.csv'))\
               .set_index(['mode', 'trial']).ix['experimentC']

    results = {}
    hyps = list(data['ipe']['C'].P_fall_smooth.columns)
    prior = util.normalize(np.zeros((1, len(hyps))), axis=1)[1]
    for (kappa0, pid), df in data['human']['C'].groupby(['kappa0', 'pid']):
        order = trials[pid]

        pfall = np.asarray(data['ipe']['C'].P_fall_smooth[hyps].ix[order])
        fall = np.asarray(data['fb']['C'].fall.ix[order][kappa0])[:, None]

        lh = np.log((fall * pfall) + ((1 - fall) * (1 - pfall)))
        prior_lh = np.vstack([prior, lh])
        posterior = util.normalize(prior_lh.cumsum(axis=0), axis=1)[1]

        res = pd.DataFrame(
            posterior, index=['prior'] + list(order), columns=hyps)
        res['trial'] = np.arange(len(order) + 1)
        res = res.set_index('trial', append=True).stack()
        res.index.names = ['stimulus', 'trial', 'hypothesis']
        res = res.reset_index('stimulus').rename(columns={0: 'logp'}).stack()
        results[(kappa0, pid)] = res

    results = pd.DataFrame.from_dict(results)
    results.columns = pd.MultiIndex.from_tuples(
        results.columns, names=['kappa0', 'pid'])
    results = results\
        .stack(['kappa0', 'pid'])\
        .unstack(2)\
        .reorder_levels(['kappa0', 'pid', 'trial', 'hypothesis'])\
        .sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
