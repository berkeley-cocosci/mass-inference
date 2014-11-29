#!/usr/bin/env python

"""
Computes average accuracy across participants for "which is heavier?" as a
function of trial. Produces a csv file with the following columns:

    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    num_mass_trials (int)
        number of trials participants responded to "which is heavier?". A value
        of -1 indicates between-subjects
    trial (int)
        trial number
    lower (float)
        lower bound of the 95% confidence interval
    median
        median of the beta distribution
    upper
        upper bound of the 95% confidence interval

"""

import util
import pandas as pd
import numpy as np


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()['C'].dropna(subset=['mass? response'])

    between_subjs = human\
        .groupby('version')\
        .get_group('I')\
        .sort(['pid', 'trial'])\
        .drop_duplicates('pid')
    between_subjs['num_mass_trials'] = -1
    responses = pd.concat([human, between_subjs])

    results = []
    for kappa in [-1.0, 1.0, 'all']:
        if kappa == 'all':
            correct = responses
        else:
            correct = responses.groupby('kappa0').get_group(kappa)

        accuracy = correct\
            .groupby(['version', 'num_mass_trials', 'trial'])['mass? correct']\
            .apply(util.beta)\
            .unstack(-1)\
            .reset_index()

        accuracy['kappa0'] = kappa
        results.append(accuracy)

    results = pd.concat(results)\
        .set_index(['version', 'kappa0', 'num_mass_trials', 'trial'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.results_path, args.seed)

