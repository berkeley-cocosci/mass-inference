#!/usr/bin/env python

import util
import pandas as pd

filename = "num_learned_by_trial.csv"


def run(data, results_path, seed):

    sp = pd\
        .read_csv(results_path.joinpath('switchpoint.csv'))\
        .set_index(['version', 'kappa0', 'pid'])
    sp.columns.name = 'trial'

    results = sp\
        .stack()\
        .groupby(level=['version', 'kappa0', 'trial'])\
        .apply(util.beta)\
        .unstack(-1)

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)