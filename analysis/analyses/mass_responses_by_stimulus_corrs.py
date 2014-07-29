#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)
    results = []

    responses = pd.read_csv(path(results_path).dirname().joinpath(
        'mass_responses_by_stimulus.csv'))
    responses = responses.set_index(
        ['version', 'species', 'kappa0', 'stimulus'])['median']

    results = {}
    for version, df in responses.groupby(level='version'):
        m = df.unstack('species')

        x = m['ipe']
        y = m['human']
        results[(version, 'IPE', 'Human')] = util.bootcorr(
            x, y, method='spearman')

        x = m['empirical']
        y = m['human']
        results[(version, 'Empirical', 'Human')] = util.bootcorr(
            x, y, method='spearman')

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['version', 'X', 'Y'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
