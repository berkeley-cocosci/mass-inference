#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "participant_fits.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    def rank(x):
        y = np.asarray(x).squeeze()
        i = np.argsort(y)[::-1]
        cols = x.columns[i]
        index = ["rank_%d" % (j + 1) for j in range(len(cols))]
        ranks = pd.Series(cols, index=index)
        return ranks

    llh = pd.read_csv(results_path.joinpath('model_log_lh.csv'))\
            .set_index(['pid', 'trial'])
    llh = llh.groupby(level='pid').sum()
    results = llh\
        .groupby(level='pid')\
        .apply(rank)

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
