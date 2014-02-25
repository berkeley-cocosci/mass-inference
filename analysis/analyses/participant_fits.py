#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "participant_fits.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = pd.read_csv(results_path.joinpath('model_log_lh.csv'))\
                .set_index(['pid', 'trial'])
    results = results.groupby(level='pid').sum()
    results['best'] = results\
        .groupby(level='pid')\
        .apply(lambda x: results.columns[np.argmax(np.asarray(x).squeeze())])

    results = results\
        .reset_index()\
        .set_index(['best', 'pid'])\
        .sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
