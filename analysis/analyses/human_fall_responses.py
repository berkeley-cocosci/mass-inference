#!/usr/bin/env python

"""
Computes the average participant judgments to "will it fall?". Produces a csv
file with the following columns:

    version (string)
        the experiment version
    block (string)
        the experiment phase
    kappa0 (float)
        the true log mass ratio
    stimulus (string)
        the name of the stimulus
    lower (float in [0, 1])
        lower bound of the 95% confidence interval
    median (float in [0, 1])
        median value across the bootstrapped means
    upper (float in [0, 1])
        upper bound of the 95% confidence interval
    N (int)
        how many samples the mean was computed over
    mean (float)
        raw mean of judgments
    stddev (float)
        standard deviation of judgments

"""

__depends__ = ["human_fall_responses_raw.csv"]
__random__ = True

import util
import pandas as pd
import numpy as np
import os


def run(dest, results_path, seed):
    np.random.seed(seed)
    data = pd.read_csv(os.path.join(results_path, 'human_fall_responses_raw.csv'))\
        .groupby(['version', 'block', 'kappa0', 'stimulus'])['fall? response']
    results = data.apply(util.bootstrap_mean).unstack(-1)
    results['mean'] = data.mean()
    results['stddev'] = data.std()
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed)

