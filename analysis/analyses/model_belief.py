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
    for lhtype in ('empirical', 'ipe'):
        hyps = [-1.0, 1.0]
        prior = util.normalize(np.zeros((1, len(hyps))), axis=1)[1]
        groups = data['human']['C'].groupby(['version', 'kappa0', 'pid'])
        for (version, kappa0, pid), df in groups:
            order = trials[pid]

            pfall = np.asarray(data[lhtype]['C'].P_fall_smooth[hyps].ix[order])
            fall = np.asarray(data['fb']['C'].fall.ix[order][kappa0])[:, None]

            lh = np.log((fall * pfall) + ((1 - fall) * (1 - pfall)))
            prior_lh = np.vstack([prior, lh])
            posterior = util.normalize(prior_lh.cumsum(axis=0), axis=1)[1]
            posterior_ind = util.normalize(prior_lh, axis=1)[1]

            # hack to handle rounding error
            if (posterior == 0).any():
                ix = np.argwhere(posterior == 0)
                rows = ix[:, 0]
                cols = ix[:, 1]
                vals = posterior[rows][:, 1 - cols]
                other = np.log(1 - np.exp(vals))
                posterior[ix] = other

            res = pd.DataFrame(
                posterior, index=['prior'] + list(order), columns=hyps)
            res['trial'] = np.arange(len(order) + 1)
            res = res.set_index('trial', append=True).stack()
            res.index.names = ['stimulus', 'trial', 'hypothesis']
            res = res.reset_index('stimulus')\
                     .rename(columns={0: 'logp'})\
                     .stack()
            results[(lhtype, 'learning', version, kappa0, pid)] = res

            res = pd.DataFrame(
                posterior_ind, index=['prior'] + list(order), columns=hyps)
            res['trial'] = np.arange(len(order) + 1)
            res = res.set_index('trial', append=True).stack()
            res.index.names = ['stimulus', 'trial', 'hypothesis']
            res = res.reset_index('stimulus')\
                     .rename(columns={0: 'logp'})\
                     .stack()
            results[(lhtype, 'static', version, kappa0, pid)] = res

            chance = res.copy().unstack(-1)
            chance.loc[:, 'logp'] = prior.flat[0]
            results[(lhtype, 'chance', version, kappa0, pid)] = chance.stack()

    results = pd.DataFrame.from_dict(results)
    results.columns = pd.MultiIndex.from_tuples(
        results.columns,
        names=['likelihood', 'model', 'version', 'kappa0', 'pid'])
    results = results\
        .stack(['likelihood', 'model', 'version', 'kappa0', 'pid'])\
        .unstack(2)\
        .reorder_levels(
            ['model', 'likelihood', 'version', 'kappa0',
             'pid', 'trial', 'hypothesis'])\
        .sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
