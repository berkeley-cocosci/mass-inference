#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def make_model_df(name, arr, rows, cols, trials):
    df = pd.DataFrame(
        arr, index=rows, columns=cols)
    df['trial'] = trials
    df = df.set_index('trial', append=True)
    df.columns.name = cols.name
    df = df\
        .stack()\
        .reset_index()\
        .rename(columns={
            0: 'logp',
            'kappa': 'hypothesis'
        })
    df['model'] = name
    return df


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_human()
    results = []

    llh = pd.read_csv(
        path(results_path).dirname().joinpath("model_belief.csv"))

    pth = path(results_path).dirname().joinpath("trial_order.csv")
    trials = pd.read_csv(pth)\
               .set_index(['mode', 'trial'])\
               .groupby(level='mode')\
               .get_group('experimentC')

    human = data['C']
    participants = human[['version', 'kappa0', 'pid']]\
        .drop_duplicates()\
        .sort(['version', 'kappa0', 'pid'])
    participants = np.asarray(participants)

    results = []
    for key, df in llh.groupby(['likelihood', 'query', 'sigma', 'phi']):
        print key
        (lh, query, sigma, phi) = key
        belief = df\
            .set_index(['kappa0', 'stimulus', 'kappa'])['llh']\
            .unstack('kappa')

        for (version, kappa0, pid) in participants:
            sys.stdout.write('.')
            sys.stdout.flush()

            # put the array in the correct trial order
            order = trials[pid].dropna()
            trial_nums = np.asarray(
                order.index.get_level_values('trial'), dtype=float)
            pbelief_df = belief.ix[kappa0].ix[order]
            rows = pbelief_df.index
            rows.name = 'stimulus'
            cols = pbelief_df.columns
            pbelief = np.asarray(pbelief_df)

            # learning model
            learning = util.normalize(np.cumsum(pbelief, axis=0), axis=1)[1]
            learning_df = make_model_df(
                'learning', learning, rows, cols, trial_nums)

            # static model
            static = util.normalize(pbelief, axis=1)[1]
            static_df = make_model_df('static', static, rows, cols, trial_nums)

            # chance model
            chance = util.normalize(np.zeros(learning.shape), axis=1)[1]
            chance_df = make_model_df('chance', chance, rows, cols, trial_nums)

            # put them all together
            models = pd.concat([static_df, learning_df, chance_df])
            models['version'] = version
            models['kappa0'] = kappa0
            models['pid'] = pid
            models['likelihood'] = lh
            models['query'] = query
            models['sigma'] = sigma
            models['phi'] = phi

            results.append(models)

        sys.stdout.write('\n')

    results = pd\
        .concat(results)\
        .set_index([
            'model', 'likelihood', 'query', 'version', 'kappa0', 'pid'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
