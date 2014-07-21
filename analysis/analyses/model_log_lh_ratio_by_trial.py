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

    llh_I_within = llh\
        .groupby('version')\
        .get_group('I')\
        .set_index(['likelihood', 'model', 'pid'])\
        .groupby(level=['likelihood', 'model', 'pid'])\
        .apply(lambda x: x.sort('trial').irow(0))\
        .reset_index()
    llh_I_within['version'] = "I (across)"

    llh_I_across = llh\
        .groupby('version')\
        .get_group('I')\
        .set_index(['version', 'model', 'likelihood', 'pid', 'trial'])['llh']\
        .unstack('trial')\
        .dropna(subset=[1])\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'llh'})
    llh_I_across['version'] = "I (within)"

    llh = llh\
        .set_index('version')\
        .drop('I', axis=0)\
        .reset_index()
    llh = pd.concat([llh, llh_I_within, llh_I_across])

    llh_trial = 2 * llh\
        .groupby('likelihood')\
        .get_group('empirical')\
        .pivot_table(rows=['version', 'model'],
                     cols='trial', values='llh',
                     aggfunc=np.sum)\
        .stack()\
        .unstack('model')

    llh_trial_H = llh_trial.groupby(level='version').get_group('H')
    llr_H = (llh_trial_H['static'] - llh_trial_H['chance'])\
        .reset_index()\
        .rename(columns={0: 'llhr'})
    llr_H['model'] = None
    llr_H.loc[llr_H['llhr'] < 0, 'model'] = 'chance'
    llr_H.loc[llr_H['llhr'] > 0, 'model'] = 'static'

    llh_trial_GI = llh_trial.drop('H')
    llr_GI = llh_trial_GI['learning'] - llh_trial_GI['static']
    llr_GI = llr_GI\
        .reset_index()\
        .rename(columns={0: 'llhr'})
    llr_GI['model'] = None
    llr_GI.loc[llr_GI['llhr'] < 0, 'model'] = 'static'
    llr_GI.loc[llr_GI['llhr'] > 0, 'model'] = 'learning'

    llr = pd.concat([llr_H, llr_GI])
    llr.loc[:, 'llhr'] = llr['llhr']
    llr['evidence'] = 'equal'
    llr.loc[np.abs(llr['llhr']) > 0, 'evidence'] = 'weak'
    llr.loc[np.abs(llr['llhr']) > 2, 'evidence'] = 'positive'
    llr.loc[np.abs(llr['llhr']) > 6, 'evidence'] = 'strong'
    llr.loc[np.abs(llr['llhr']) > 10, 'evidence'] = 'very strong'

    results = llr.set_index(['version', 'trial'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
