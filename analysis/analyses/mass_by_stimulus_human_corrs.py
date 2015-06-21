#!/usr/bin/env python

"""
Computes the pearson correlation between model and human judgments on "which is
heavier?" for each version of the experiment. Produces a csv file with the
following columns:

    version (string)
        the experiment version
    lower (float)
        lower bound of the 95% confidence interval
    median (float)
        median of the bootstrap distribution
    upper (float)
        upper bound of the 95% confidence interval

"""

__depends__ = ["human_mass_responses_by_stimulus_raw.csv", "human_mass_accuracy_by_stimulus_raw.csv"]
__random__ = True

import os
import util
import pandas as pd
import numpy as np
import scipy.stats


def nanbeta(arr, axis=1):
    alpha = np.nansum(arr, axis=axis) + 0.5
    beta = np.nansum(1 - arr, axis=axis) + 0.5
    return alpha / (alpha + beta)


def within_corr(arr):
    ix = np.arange(arr.shape[1])
    np.random.shuffle(ix)
    half1 = nanbeta(arr[:, ix[:len(ix)/2]], axis=1)
    half2 = nanbeta(arr[:, ix[len(ix)/2:]], axis=1)
    return scipy.stats.pearsonr(half1, half2)[0]


def run(dest, results_path, seed):
    np.random.seed(seed)

    responses = pd\
        .read_csv(os.path.join(results_path, "human_mass_responses_by_stimulus_raw.csv"))\
        .groupby('version')\
        .get_group('H')\
        .set_index(['stimulus', 'kappa0', 'pid'])['mass? response']\
        .unstack('pid')

    accuracy = pd\
        .read_csv(os.path.join(results_path, "human_mass_accuracy_by_stimulus_raw.csv"))\
        .groupby('version')\
        .get_group('H')\
        .set_index(['stimulus', 'kappa0', 'pid'])['mass? correct']\
        .unstack('pid')

    results = {}

    raw = np.asarray(responses)
    corrs = np.array([within_corr(raw) for _ in range(10000)])
    results['mass? response'] = pd.Series(
        np.percentile(corrs, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'])

    raw = np.asarray(accuracy)
    corrs = np.array([within_corr(raw) for _ in range(10000)])
    results['mass? correct'] = pd.Series(
        np.percentile(corrs, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'])

    results = pd.DataFrame(results).T
    results.index.name = 'judgment'
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed)
