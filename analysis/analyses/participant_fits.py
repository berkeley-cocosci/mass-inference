#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "participant_fits.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    def rank(x):
        y = np.asarray(x).squeeze()
        i = np.argsort(y)[::-1]
        cols = x.columns[i]
        index = np.arange(len(cols), dtype=int)
        ranks = pd.Series(cols, index=index)
        return ranks

    llh = pd.read_csv(results_path.joinpath('model_log_lh.csv'))\
            .groupby('likelihood')\
            .get_group('empirical')\
            .set_index(['version', 'pid', 'trial', 'model'])['llh']\
            .unstack('model')

    results = llh\
        .groupby(level=['version', 'pid'])\
        .sum()\
        .groupby(level=['version', 'pid'])\
        .apply(rank)\
        .stack()\
        .reset_index()\
        .rename(columns={
            'level_2': 'rank',
            0: 'model'
        })\
        .set_index(['version', 'pid'])

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
