#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
import scipy.stats
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    cols = ['version', 'kappa0', 'num_mass_trials']
    means = pd.read_csv(path(results_path).dirname().joinpath(
        "mass_accuracy_by_trial.csv"))
    means = means\
        .groupby(['species', 'class'])\
        .get_group(('human', 'chance'))\
        .set_index(cols)[['trial', 'median']]

    def corr(df):
        x = np.asarray(df['trial'])
        y = np.asarray(df['median'])
        samps = np.random.rand(10000, 80, y.size) < y
        samps = samps.mean(axis=1)
        corrs = np.array([scipy.stats.spearmanr(x, s)[0] for s in samps])
        stats = pd.Series(
            np.percentile(corrs, [2.5, 50, 97.5]),
            index=['lower', 'median', 'upper'])
        stats.name = df.name
        return stats

    results = means.groupby(level=cols).apply(corr)

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
