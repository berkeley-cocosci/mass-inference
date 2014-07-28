#!/usr/bin/env python

import sys
import util
import numpy as np


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()
    results = []

    results = human['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .groupby(['version', 'num_mass_trials', 'pid'])['mass? correct']\
        .mean()\
        .to_frame()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
