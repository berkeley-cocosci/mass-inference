#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
import scipy.stats
from path import path
from util import exponentiated_luce_choice as elc
from scipy.optimize import curve_fit


def run(results_path, seed):
    np.random.seed(seed)

    human = util\
        .load_human()['C']\
        .groupby('version')\
        .get_group('H')\
        .dropna(axis=0, subset=['mass? response'])\
        .groupby(['kappa0', 'stimulus'])['mass? correct']\
        .mean()\

    model_belief = pd.read_csv(path(results_path).dirname().joinpath(
        'model_belief_agg_all_params.csv'))

    model = model_belief\
        .groupby(['likelihood', 'model', 'version'])\
        .get_group(('ipe', 'static', 'H'))\
        .groupby(['sigma', 'phi', 'kappa0', 'stimulus'])['p correct']\
        .mean()\
        .groupby(lambda x: x[-1] == 'prior')\
        .get_group(False)

    results = {}
    for (sigma, phi), model_df in model.groupby(level=['sigma', 'phi']):
        x0 = np.asarray(model_df)
        y = np.asarray(human)
        g = curve_fit(elc, x0, y)[0]
        x = elc(x0, g)
        corr = scipy.stats.pearsonr(x, y)[0]
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
