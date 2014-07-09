#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_responses_by_stimulus_corrs.csv"


def run(results_path, seed):
    np.random.seed(seed)
    results = []

    responses = pd\
        .read_csv(results_path.joinpath('mass_responses_by_stimulus.csv'))\
        .set_index(['version', 'species', 'kappa0', 'stimulus'])['median']

    results = {}
    for version, df in responses.groupby(level='version'):
        m = df.unstack('species')

        x = m['ipe']
        y = m['human']
        results[(version, 'IPE', 'Human')] = util.bootcorr(x, y)

        x = m['empirical']
        y = m['human']
        results[(version, 'Empirical IPE', 'Human')] = util.bootcorr(x, y)

    results = pd.DataFrame.from_dict(results, orient='index')
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['version', 'X', 'Y'])

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
