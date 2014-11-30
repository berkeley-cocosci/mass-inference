#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


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

    query = util.get_query()
    belief = pd.read_csv(path(results_path).dirname().joinpath(
        'single_model_belief.csv'))
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

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
