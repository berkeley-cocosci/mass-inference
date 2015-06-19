#!/usr/bin/env python

"""
Computes average model probability of saying r=10 for "which is heavier?".
Produces a csv file with the following columns:

    likelihood (string)
        the likelihood name (e.g., ipe, ipe_cf, empirical, empirical_cf)
    counterfactual (bool)
        whether the counterfactual likelihood was used
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

"""

__depends__ = ["model_likelihood.csv"]

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path):

    llh = pd.read_csv(os.path.join(results_path, 'model_likelihood.csv'))

    result = llh\
        .groupby('hypothesis')\
        .get_group(1)\
        .drop('hypothesis', axis=1)\
        .set_index(['likelihood', 'counterfactual', 'stimulus', 'kappa0'])\
        .sortlevel()
    result = np.exp(result)

    result.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
