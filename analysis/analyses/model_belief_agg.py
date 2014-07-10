#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_belief_agg.csv"


def run(results_path, seed):
    np.random.seed(seed)

    belief = pd.read_csv(
        results_path.joinpath("model_belief_agg_all_params.csv"))

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
        .set_index(['model', 'likelihood', 'version'])

    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
