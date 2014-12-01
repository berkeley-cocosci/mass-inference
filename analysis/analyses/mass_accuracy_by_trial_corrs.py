#!/usr/bin/env python

__depends__ = ["human_mass_accuracy_by_trial.csv"]

import os
import util
import pandas as pd
import numpy as np
import scipy.stats


def corr(df):
    x = np.asarray(df['trial'])
    y = np.asarray(df['median'])
    N = np.asarray(df['N'])
    if y.size == 1:
        stats = pd.Series(
            [np.nan, np.nan, np.nan],
            index=['lower', 'median', 'upper'])
    else:
        f = lambda i: (np.random.rand(10000, N[i]) < y[i]).mean(axis=1)
        samps = np.array([f(i) for i in range(y.size)]).T
        corrs = np.array([scipy.stats.spearmanr(x, s)[0] for s in samps])
        stats = pd.Series(
            np.percentile(corrs, [2.5, 50, 97.5]),
            index=['lower', 'median', 'upper'])
    stats.name = df.name
    return stats


def run(dest, results_path, seed):
    np.random.seed(seed)

    # load in human data
    human = pd.read_csv(os.path.join(results_path, "human_mass_accuracy_by_trial.csv"))

    # compute correlations
    results = human\
        .groupby('kappa0')\
        .get_group('all')\
        .groupby(['version', 'num_mass_trials'])\
        .apply(corr)

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals(), seed=True)
    args = parser.parse_args()
    run(args.dest, args.results_path, args.seed)

