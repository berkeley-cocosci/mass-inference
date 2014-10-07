#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)
    results = []

    accuracy = pd.read_csv(path(results_path).dirname().joinpath(
        'mass_accuracy_by_stimulus.csv'))
    accuracy = accuracy.set_index(
        ['version', 'species', 'kappa0', 'stimulus'])['median']

    results = {}
    for version, df in accuracy.groupby(level='version'):
        m = df.unstack('species')

        x = m['ipe']
        y = m['human']
        results[(version, 'IPE', 'Human')] = util.bootcorr(
            x, y, method='pearson')

        x = m['empirical']
        y = m['human']
        results[(version, 'Empirical', 'Human')] = util.bootcorr(
            x, y, method='pearson')

    results = pd.DataFrame.from_dict(results).T
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['version', 'X', 'Y'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
