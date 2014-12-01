#!/usr/bin/env python

"""
Computes average model probability of saying r=10 for "which is heavier?".
Produces a csv file with the following columns:

    likelihood (string)
        the likelihood name (e.g., ipe, ipe_cf, empirical, empirical_cf)
    counterfactual (bool)
        whether the counterfactual likelihood was used
    model (string)
        the model version (e.g., static, learning)
    fitted (bool)
        whether the model was fit to human responses
    version (string)
        experiment version
    kappa0 (float)
        true log mass ratio
    stimulus (string)
        stimulus name
    lower (float)
        lower bound of the 95% confidence interval
    median (float)
        median of the bootstrap distribution
    upper (float)
        upper bound of the 95% confidence interval
    N (int)
        how many samples the mean was computed over

"""

__depends__ = ["single_model_belief.csv"]
__random__ = True

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path, seed):
    np.random.seed(seed)
    model_belief = pd.read_csv(os.path.join(results_path, 'single_model_belief.csv'))
    cols = ['likelihood', 'counterfactual', 'model', 'fitted', 'version', 'kappa0', 'stimulus']
    results = model_belief\
        .groupby(cols)['p']\
        .apply(util.bootstrap_mean)\
        .unstack(-1)
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed)
