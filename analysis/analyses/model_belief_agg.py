#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_belief_agg.csv"


def run(data, results_path, seed):
    np.random.seed(seed)

    results = pd.read_csv(
        results_path.joinpath("model_belief.csv"))
    results['p'] = np.exp(results['logp'])

    belief = results\
        .set_index(['kappa0', 'trial', 'pid', 'hypothesis'])['p']\
        .unstack('hypothesis')

    mask = np.zeros(len(belief.columns))
    mask[belief.columns > 0] = 1
    mask[belief.columns == 0] = 0.5

    belief = (belief * mask)\
        .sum(axis=1)\
        .reset_index()\
        .rename(columns={0: 'p'})

    ix = belief['kappa0'] < 0
    p = belief['p'].copy()
    p[ix] = 1 - p[ix]
    belief.loc[:, 'p'] = p
    belief = belief.set_index(['kappa0', 'trial', 'pid'])

    pth = results_path.joinpath(filename)
    belief.to_csv(pth)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
