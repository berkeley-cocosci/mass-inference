#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    def rank(x):
        y = np.asarray(x).squeeze()
        i = np.argsort(y)[::-1]
        cols = x.columns[i]
        index = np.arange(len(cols), dtype=int)
        ranks = pd.Series(cols, index=index)
        return ranks

    llh = pd.read_csv(path(results_path).dirname().joinpath(
        'model_log_lh_all.csv'))
    llh = llh\
        .groupby('likelihood')\
        .get_group('empirical')\
        .set_index(['version', 'pid', 'trial', 'model'])['llh']\
        .unstack('model')

    llh_sum = llh.groupby(level=['version', 'pid']).sum()

    results = llh_sum\
        .groupby(level=['version', 'pid'])\
        .apply(rank)\
        .stack()\
        .reset_index()\
        .rename(columns={
            'level_2': 'rank',
            0: 'model'
        })\
        .set_index(['version', 'pid', 'model'])\
        .sortlevel()

    results.loc[:, 'llh'] = llh_sum.stack()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
