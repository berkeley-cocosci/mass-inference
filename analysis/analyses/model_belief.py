#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from itertools import product as iproduct
from path import path


def make_belief(pfall, fall, hyps, index, prior, trial_nums):
    lh = np.log((fall * pfall) + ((1 - fall) * (1 - pfall)))
    shape = list(lh.shape)
    shape[-2] += 1
    prior_lh = np.empty(shape)
    prior_lh[..., 0, :] = prior
    prior_lh[..., 1:, :] = lh
    posterior = util.normalize(prior_lh.cumsum(axis=-2), axis=-1)[1]
    posterior = posterior.reshape((-1, len(hyps)))
    posterior_ind = util.normalize(prior_lh, axis=-1)[1]
    posterior_ind = posterior_ind.reshape((-1, len(hyps)))

    # hack to handle rounding error
    if (posterior == 0).any():
        ix = np.argwhere(posterior == 0)
        ix2 = ix.copy()
        ix2[:, -1] = 1 - ix2[:, -1]
        vals = posterior[tuple(ix2.T)]
        other = np.log(1 - np.exp(vals))
        posterior[tuple(ix.T)] = other

    # make the list of trials
    trials = np.empty(prior_lh.shape[:-1])
    trials[..., 0] = 0
    trials[..., 1:] = trial_nums
    trials = trials.ravel()

    # create the learning model
    learning = pd.DataFrame(
        posterior, index=index, columns=hyps)
    learning['trial'] = trials
    learning = learning.set_index('trial', append=True)
    learning.columns.name = 'hypothesis'
    learning = learning\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'logp'})
    learning['model'] = 'learning'

    reverse_learning = learning.copy()
    reverse_learning.loc[:, 'logp'] = np.log(1 - np.exp(learning['logp']))
    reverse_learning.loc[:, 'model'] = 'reverse_learning'

    # create the static model
    static = pd.DataFrame(
        posterior_ind, index=index, columns=hyps)
    static['trial'] = trials
    static = static.set_index('trial', append=True)
    static.columns.name = 'hypothesis'
    static = static\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'logp'})
    static['model'] = 'static'

    reverse_static = static.copy()
    reverse_static.loc[:, 'logp'] = np.log(1 - np.exp(static['logp']))
    reverse_static.loc[:, 'model'] = 'reverse_static'

    # create the chance model
    chance = static.copy()
    chance.loc[:, 'logp'] = prior.flat[0]
    chance['model'] = 'chance'

    models = pd.concat([
        learning, reverse_learning, static, reverse_static, chance])

    return models


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_all()
    results = []

    pth = path(results_path).dirname().joinpath("trial_order.csv")
    trials = pd.read_csv(pth)\
               .set_index(['mode', 'trial'])\
               .groupby(level='mode')\
               .get_group('experimentC')

    human = data['human']['C']
    participants = human[['version', 'kappa0', 'pid']]\
        .drop_duplicates()\
        .sort(['version', 'kappa0', 'pid'])
    participants = np.asarray(participants)

    hyps = [-1.0, 1.0]
    prior = util.normalize(np.zeros((1, len(hyps))), axis=1)[1]

    ipe = data['ipe']['C']\
        .P_fall_mean_all[hyps]\
        .reorder_levels(['stimulus', 'sigma', 'phi'])\
        .sortlevel()\
        .reset_index(['sigma', 'phi'])
    empirical = data['empirical']['C'].P_fall_mean[hyps]
    fb = data['fb']['C'].fall

    results = []
    for (version, kappa0, pid) in participants:
        print version, kappa0, pid
        order = trials[pid].dropna()
        trial_nums = np.asarray(
            order.index.get_level_values('trial'), dtype=float)
        fall = np.asarray(fb.ix[order][kappa0])[:, None]

        pfall = np.asarray(empirical.ix[order])
        index = pd.Index(['prior'] + list(order), name='stimulus')
        belief = make_belief(pfall, fall, hyps, index, prior, trial_nums)
        belief['version'] = version
        belief['kappa0'] = kappa0
        belief['pid'] = pid
        belief['likelihood'] = 'empirical'
        belief['sigma'] = np.nan
        belief['phi'] = np.nan
        results.append(belief)

        ipe_df = ipe\
            .ix[order]\
            .set_index(['sigma', 'phi'], append=True)
        ipe_groups = ipe_df.groupby(level=['sigma', 'phi'])
        keys, pfall = zip(*[(x[0], np.asarray(x[1])) for x in ipe_groups])
        pfall = np.array(pfall)

        index = list(iproduct(keys, ['prior'] + list(order)))
        index = [(x[0][0], x[0][1], x[1]) for x in index]
        index = pd.MultiIndex.from_tuples(index)
        index.names = ['sigma', 'phi', 'stimulus']
        belief = make_belief(pfall, fall, hyps, index, prior, trial_nums)
        belief['version'] = version
        belief['kappa0'] = kappa0
        belief['pid'] = pid
        belief['likelihood'] = 'ipe'
        results.append(belief)

    results = pd\
        .concat(results)\
        .set_index(['model', 'likelihood', 'version', 'kappa0',
                    'pid', 'trial', 'hypothesis'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
