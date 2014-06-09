#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_log_lh.csv"


def run(data, results_path, version, seed):
    np.random.seed(seed)
    results = {}

    h = data['human']['C'].pivot('pid', 'trial', 'mass? correct')

    def llh(df):
        m = df.pivot('pid', 'trial', 'p')
        lh = np.log(((m * h) + ((1 - m) * (1 - h))).dropna(axis=1))
        lh = lh.stack()
        lh.name = df.name
        return lh

    belief = pd.read_csv(results_path.joinpath(
        version, 'model_belief_agg.csv'))
    results = belief\
        .groupby(['likelihood', 'model'])\
        .apply(llh)\
        .stack()\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'llh'})\
        .set_index(['likelihood', 'model'])

    pth = results_path.joinpath(version, filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
