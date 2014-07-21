#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    llh = pd.read_csv(path(results_path).dirname().joinpath(
        'model_log_lh.csv'))

    def add_num_trials(df):
        df['num_trials'] = len(df['trial'].unique())
        return df

    llh = llh\
        .groupby('pid')\
        .apply(add_num_trials)\
        .groupby('likelihood')\
        .get_group('empirical')\
        .groupby(['version', 'model', 'num_trials', 'pid'])[['llh']]\
        .sum()\
        .reset_index()\
        .set_index(['version', 'num_trials', 'pid', 'model'])['llh']\
        .unstack('model')

    llh_H = llh.groupby(level='version').get_group('H')
    llr_H = (llh_H['static'] - llh_H['chance'])\
        .reset_index()\
        .rename(columns={0: 'llhr'})
    llr_H['model'] = 'equal'
    llr_H.loc[llr_H['llhr'] < 0, 'model'] = 'chance'
    llr_H.loc[llr_H['llhr'] > 0, 'model'] = 'static'

    llh_GI = llh.drop('H')
    llr_GI = llh_GI['learning'] - llh_GI['static']
    llr_GI = llr_GI\
        .reset_index()\
        .rename(columns={0: 'llhr'})
    llr_GI['model'] = 'equal'
    llr_GI.loc[llr_GI['llhr'] < 0, 'model'] = 'static'
    llr_GI.loc[llr_GI['llhr'] > 0, 'model'] = 'learning'

    llr = pd.concat([llr_H, llr_GI])
    llr.loc[:, 'llhr'] = llr['llhr']
    llr['evidence'] = 'equal'
    llr.loc[np.abs(llr['llhr'] > 0), 'evidence'] = 'weak'
    llr.loc[np.abs(llr['llhr'] > 1), 'evidence'] = 'positive'
    llr.loc[np.abs(llr['llhr'] > 3), 'evidence'] = 'strong'
    llr.loc[np.abs(llr['llhr'] > 5), 'evidence'] = 'very strong'

    results = llr.set_index(['version', 'num_trials', 'pid', 'model'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
