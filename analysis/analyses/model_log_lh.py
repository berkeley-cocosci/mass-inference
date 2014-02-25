#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_log_lh.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    results = {}

    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))
    h = data['human']['C'].pivot('pid', 'trial', 'mass? correct')
    for lhtype, df in belief.groupby('likelihood'):
        m = df.pivot('pid', 'trial', 'p')
        lh = np.log(((m * h) + ((1 - m) * (1 - h))).dropna(axis=1))
        results[lhtype] = lh.stack()

    results = pd.DataFrame(results)

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
