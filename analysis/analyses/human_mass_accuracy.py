#!/usr/bin/env python

"""
Computes overall participant accuracy for each version of the experiment for
each separate mass ratio, and for both mass ratios combined. Produces a csv file
with the following columns:

    version (string)
        the experiment version
    kappa0 (float)
        true log mass ratio
    lower (float)
        lower bound of the 95% confidence interval
    median
        median of the beta distribution
    upper
        upper bound of the 95% confidence interval
    N (int)
        how many samples the mean was computed over

"""

__depends__ = ["human"]

import util
import pandas as pd
import numpy as np


def run(dest, data_path, seed):
    np.random.seed(seed)
    human = util.load_human(data_path)['C'].dropna(subset=['mass? response'])

    results = []
    for kappa in [-1.0, 1.0, 'all']:
        if kappa == 'all':
            correct = human
        else:
            correct = human.groupby('kappa0').get_group(kappa)

        accuracy = correct\
            .groupby('version')['mass? correct']\
            .apply(util.beta)\
            .unstack(-1)\
            .reset_index()

        accuracy['kappa0'] = kappa
        results.append(accuracy)

    results = pd.concat(results)\
        .set_index(['version', 'kappa0'])\
        .sortlevel()

    results.to_csv(dest)

if __name__ == "__main__":
    parser = util.default_argparser(locals(), seed=True)
    args = parser.parse_args()
    run(args.dest, args.data_path, args.seed)
