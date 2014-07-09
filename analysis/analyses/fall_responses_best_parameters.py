#!/usr/bin/env python

import util
import pandas as pd
import numpy as np
import scipy.stats

filename = "fall_responses_best_parameters.csv"


def run(results_path, seed):
    np.random.seed(seed)
    human = util\
        .load_human()['B']\
        .set_index(['version', 'stimulus', 'kappa0', 'pid'])['fall? response']\
        .groupby(level=['stimulus', 'kappa0'])\
        .mean()

    ipe = util.load_model()[0]['B']
    model = ipe.data.copy().drop(['nfell'], axis=1)
    model['unstable'] = (ipe.data['nfell'] > 4).astype(float)
    model = model\
        .set_index(['sigma', 'phi', 'stimulus'])\
        .groupby(level=['sigma', 'phi', 'stimulus'])\
        .apply(ipe._sample_kappa_mean)[[-1.0, 1.0]]\
        .stack()

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

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
