#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_log_lh.csv"


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()
    results = {}

    h = human['C']\
        .set_index(['version', 'pid', 'trial'])['mass? correct']\
        .unstack('trial')

    def llh(df):
        m = df.reset_index(['likelihood', 'model'], drop=True)
        lh = np.log(((m * h) + ((1 - m) * (1 - h))))
        lh = lh.stack()
        lh.name = df.name
        return lh

    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))
    results = belief\
        .set_index([
            'likelihood', 'model', 'version', 'pid', 'trial'])['p correct']\
        .unstack('trial')
    results = results\
        .groupby(level=['likelihood', 'model'])\
        .apply(llh)\
        .stack()\
        .stack()\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'llh'})\
        .set_index(['version', 'likelihood', 'model', 'pid'])\
        .sort()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
