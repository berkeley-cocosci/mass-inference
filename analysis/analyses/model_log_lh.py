#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    llh = pd.read_csv(path(results_path).dirname().joinpath(
        'model_log_lh_all.csv'))
    fits = pd.read_csv(path(results_path).dirname().joinpath(
        'participant_fits.csv'))

    learning_fits = fits[fits['version'] != 'H']
    models = learning_fits.pivot('pid', 'model', 'llh')
    llr = models['chance'] - models['static']
    # only mark pids as bad if there is positive non-negligible
    # evidence for chance... that is, don't exclude pids if they are
    # only very slightly in favor of chance
    bad_pids = list(llr[llr > 1].index)

    results = llh\
        .set_index('pid')\
        .drop(bad_pids)\
        .reset_index()\
        .set_index(['version', 'likelihood', 'model'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
