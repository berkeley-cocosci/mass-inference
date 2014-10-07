#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np


def run(results_path, seed):
    def find_switchpoint(df):
        version, kappa0, pid = df.name

        df = df.dropna(axis=1)
        arr = np.asarray(df).copy().ravel()

        eq = np.nonzero(~(arr.astype('bool')))[0]
        if eq.size == 0:
            idx = 0
        else:
            idx = eq[-1] + 1

        new_arr = np.empty(arr.shape)
        new_arr[:idx] = False
        new_arr[idx:] = True

        new_df = pd.DataFrame(
            new_arr[None], index=df.index, columns=df.columns)

        return new_df

    np.random.seed(seed)
    human = util.load_human()
    results = human['C']\
        .pivot_table(
            index=['version', 'kappa0', 'pid'],
            columns='trial',
            values='mass? correct')\
        .groupby(level=['version', 'kappa0', 'pid'])\
        .apply(find_switchpoint)\
        .dropna(axis=0, how='all')

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
