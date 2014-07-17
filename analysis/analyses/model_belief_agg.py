#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    belief = pd.read_csv(path(results_path).dirname().joinpath(
        "model_belief_agg_all_params.csv"))

    sigma, phi = util.get_params()

    empirical = belief\
        .groupby('likelihood')\
        .get_group('empirical')\
        .drop(['sigma', 'phi'], axis=1)
    ipe = belief\
        .groupby(['likelihood', 'sigma', 'phi'])\
        .get_group(('ipe', sigma, phi))\
        .drop(['sigma', 'phi'], axis=1)

    results = pd\
        .concat([empirical, ipe])\
        .set_index(['model', 'likelihood', 'version', 'trial'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
