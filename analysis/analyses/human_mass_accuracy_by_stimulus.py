#!/usr/bin/env python

"""
Computes average participant accuracy to "which is heavier?". Produces a csv
file with the following columns:

    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    stimulus (string)
        stimulus name
    lower (float)
        lower bound of the 95% confidence interval
    median (float)
        median of the beta distribution
    upper (float)
        upper bound of the 95% confidence interval
    N (int)
        how many samples the mean was computed over

"""

__depends__ = ["human_mass_accuracy_by_stimulus_raw.csv"]
__random__ = True

import util
import numpy as np
import os
import pandas as pd


def run(dest, results_path, seed):
    np.random.seed(seed)
    human = pd.read_csv(os.path.join(results_path, "human_mass_accuracy_by_stimulus_raw.csv"))
    results = human\
        .groupby(['version', 'kappa0', 'stimulus'])['mass? correct']\
        .apply(util.bootstrap_mean)\
        .unstack(-1)\
        .reset_index()\
        .set_index(['version', 'kappa0', 'stimulus'])\
        .sortlevel()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed)
