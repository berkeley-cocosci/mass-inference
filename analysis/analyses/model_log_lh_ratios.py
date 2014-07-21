#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    llh_trial = pd.read_csv(path(results_path).dirname().joinpath(
        'model_log_lh_ratio_by_trial.csv'))
    llh_participant = pd.read_csv(path(results_path).dirname().joinpath(
        'model_log_lh_ratio_by_participant.csv'))

    llh_across = llh_trial.pivot(
        'trial', 'version', 'llhr').sum()[['I (across)']]
    llh_across.index = pd.MultiIndex.from_tuples(
        [('I', -1)], names=['model', 'num_trials'])

    results = llh_participant\
        .groupby(['version', 'num_trials'])['llhr']\
        .sum()\
        .append(llh_across)\
        .reset_index()\
        .rename(columns={0: 'llhr'})\
        .set_index(['version', 'num_trials'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
