#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
import scipy.stats


def run(results_path, seed):
    np.random.seed(seed)
    human = util\
        .load_human()['B']\
        .set_index(['version', 'stimulus', 'kappa0', 'pid'])['fall? response']\
        .groupby(level=['stimulus', 'kappa0'])\
        .mean()

    ipe = util.load_model()[0]['B']
    model = ipe.P_fall_mean_all[[-1.0, 1.0]].stack()

    results = {}
    for (sigma, phi), model_df in model.groupby(level=['sigma', 'phi']):
        corr = scipy.stats.pearsonr(model_df, human)[0]
        results[(sigma, phi)] = corr

    results = pd.Series(results)
    results.index = pd.MultiIndex.from_tuples(results.index)
    results.index.names = ['sigma', 'phi']
    results = results\
        .reset_index(['phi'])\
        .rename(columns={0: 'pearsonr'})

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
