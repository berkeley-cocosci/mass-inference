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

"""

import util
import numpy as np


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()['C']\
          .dropna(axis=0, subset=['mass? response'])

    results = human\
        .groupby(['version', 'kappa0', 'stimulus'])['mass? correct']\
        .apply(util.beta)\
        .unstack(-1)\
        .reset_index()\
        .set_index(['version', 'kappa0', 'stimulus'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.results_path, args.seed)
