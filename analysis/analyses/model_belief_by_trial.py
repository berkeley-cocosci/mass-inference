#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def make_model_df(name, df, arr, trials):
    new_df = df.copy()
    new_df[:] = arr
    new_df['trial'] = trials
    new_df = new_df.set_index('trial', append=True)
    new_df.columns.name = 'hypothesis'
    new_df = new_df\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'logp'})
    new_df['model'] = name
    return new_df


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_human()
    results = []

    llh = pd.read_csv(
        path(results_path).dirname().joinpath("model_belief.csv"))
    llh = llh.set_index('kappa0')

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
    for (version, kappa0, pid) in participants:
        print version, kappa0, pid

        order = trials[pid].dropna()

        df = llh\
            .ix[kappa0]\
            .set_index(['stimulus', 'likelihood', 'query',
                        'sigma', 'phi', 'kappa'])['llh']\
            .unstack('kappa')\
            .reset_index()\
            .set_index('stimulus')\
            .ix[order]\
            .set_index(['likelihood', 'query', 'sigma', 'phi'], append=True)

        trial_nums = order.copy()
        trial_nums.sort()
        trial_nums = np.asarray(
            trial_nums.index.get_level_values('trial'), dtype=int) - 1
        trial_nums = trial_nums[df.index.labels[0]] + 1

        # static model
        static = util.normalize(np.asarray(df), axis=1)[1]
        static_df = make_model_df('static', df, static, trial_nums)

        # chance model
        chance = util.normalize(np.zeros(df.shape), axis=1)[1]
        chance_df = make_model_df('chance', df, chance, trial_nums)

        # learning model
        cumsum = df\
            .groupby(level=['likelihood', 'query', 'sigma', 'phi'])\
            .apply(np.cumsum)
        learning = util.normalize(np.asarray(cumsum), axis=1)[1]
        learning_df = make_model_df('learning', df, learning, trial_nums)

        # put them all together
        models = pd.concat([static_df, chance_df, learning_df])
        models['version'] = version
        models['kappa0'] = kappa0
        models['pid'] = pid

        results.append(models)

    results = pd\
        .concat(results)\
        .set_index([
            'model', 'likelihood', 'query', 'version', 'kappa0', 'pid'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
