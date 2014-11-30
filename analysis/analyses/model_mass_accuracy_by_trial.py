#!/usr/bin/env python

"""
Computes average model accuracy for "which is heavier?" on a per-trial basis.
Produces a csv file with the following columns:

    likelihood (string)
        the likelihood name (e.g., ipe, ipe_cf, empirical, empirical_cf)
    model (string)
        the model version (e.g., static, learning)
    fitted (bool)
        whether the model was fit to human responses
    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    num_mass_trials (int)
        number of mass trials (-1 indicates between-subjects)
    trial (int)
        trial number
    lower (float)
        lower bound of the 95% confidence interval
    median (float)
        median of the bootstrap distribution
    upper (float)
        upper bound of the 95% confidence interval

"""

__depends__ = ["human", "single_model_belief.csv"]

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path, data_path, seed):
    np.random.seed(seed)
    human = util.load_human(data_path)['C'].dropna(subset=['mass? response'])\
        .set_index(['version', 'kappa0', 'pid', 'trial'])\
        .sortlevel()

    model = pd.read_csv(os.path.join(results_path, 'single_model_belief.csv'))\
        .set_index(['version', 'kappa0', 'pid', 'trial'])\
        .sortlevel()
    model['num_mass_trials'] = human['num_mass_trials']
    model = model.reset_index().dropna(subset=['num_mass_trials'])

    between_subjs = model\
        .groupby('version')\
        .get_group('I')\
        .sort(['pid', 'trial'])\
        .drop_duplicates(['likelihood', 'model', 'fitted', 'pid'])
    between_subjs['num_mass_trials'] = -1
    responses = pd.concat([model, between_subjs])

    results = []
    for kappa in [-1.0, 1.0, 'all']:
        if kappa == 'all':
            correct = responses
        else:
            correct = responses.groupby('kappa0').get_group(kappa)

        cols = ['likelihood', 'model', 'fitted', 'version', 'num_mass_trials', 'trial']
        accuracy = correct\
            .groupby(cols)['p correct']\
            .apply(util.bootstrap_mean)\
            .unstack(-1)\
            .reset_index()

        accuracy['kappa0'] = kappa
        results.append(accuracy)

    results = pd.concat(results)\
        .set_index(['likelihood', 'model', 'fitted'])\
        .sortlevel()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals(), seed=True)
    args = parser.parse_args()
    run(args.dest, args.results_path, args.data_path, args.seed)
