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

__depends__ = ["human"]

import util
import numpy as np


def run(dest, data_path, seed):
    np.random.seed(seed)
    human = util.load_human(data_path)['C']\
          .dropna(axis=0, subset=['mass? response'])

    results = human\
        .groupby(['version', 'kappa0', 'stimulus'])['mass? correct']\
        .apply(util.beta)\
        .unstack(-1)

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals(), seed=True)
    args = parser.parse_args()
    run(args.dest, args.data_path, args.seed)
