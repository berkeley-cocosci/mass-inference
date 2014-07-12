#!/usr/bin/env python

import sys
import util
import numpy as np

filename = "num_chance.csv"


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()
    results = []

    groups = human['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .groupby('version')\
        .get_group('H')\
        .groupby(['kappa0', 'stimulus'])['mass? correct']
    alpha = 0.05 / len(groups.groups)
    results = groups\
        .apply(util.beta, [alpha])\
        .unstack(-1) <= 0.5

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
