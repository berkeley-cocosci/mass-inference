#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "switchpoint.csv"


def run(data, results_path, seed):
    def find_switchpoint(df):
        kappa0, pid = df.name
        arr = np.asarray(df).copy().squeeze()

        last = arr[-1]
        if (kappa0 < 0 and last == 1) or (kappa0 > 0 and last == 0):
            new_arr = np.zeros(arr.shape)
        else:
            eq = np.nonzero(arr != last)[0]
            if eq.size == 0:
                idx = 0
            elif eq[-1] == (arr.size - 2):
                idx = arr.size
            else:
                idx = eq[-1] + 1

            new_arr = np.empty(arr.shape)
            new_arr[:idx] = False
            new_arr[idx:] = True

        new_df = pd.DataFrame(
            new_arr[None], index=df.index, columns=df.columns)
        return new_df

    np.random.seed(seed)
    results = data['human']['C']\
        .pivot_table(
            rows=['kappa0', 'pid'],
            cols='trial',
            values='mass? response')\
        .dropna(axis=1)\
        .groupby(level=['kappa0', 'pid'])\
        .apply(find_switchpoint)

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
