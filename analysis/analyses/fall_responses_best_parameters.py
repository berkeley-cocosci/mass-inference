#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)
    human = pd.read_csv(path(results_path).dirname().joinpath(
        "fall_responses.csv"))

    human = human\
        .set_index(['version', 'block', 'species', 'stimulus', 'kappa0'])\
        .ix[('GH', 'B', 'human')]['median']\
        .sortlevel()

    ipe = util.load_model()[0]['B']
    model = ipe.P_fall_mean_all[[-1.0, 1.0]].stack()

    results = []
    for (sigma, phi), model_df in model.groupby(level=['sigma', 'phi']):
        x = model_df.reset_index(['sigma', 'phi'], drop=True)
        err = (x - human) ** 2
        err.index = model_df.index
        results.append(err)

    results = pd\
        .concat(results)\
        .reset_index(['sigma', 'phi'])\
        .rename(columns={0: 'sqerr'})

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
