#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def agg_belief(df):
    print df.name

    belief = df\
        .reset_index()\
        .set_index(['pid', 'stimulus', 'trial', 'hypothesis'])['p']\
        .unstack(-1)

    mask = np.zeros(len(belief.columns))
    mask[belief.columns > 0] = 1
    mask[belief.columns == 0] = 0.5

    belief = (belief * mask)\
        .sum(axis=1)\
        .reset_index()\
        .rename(columns={0: 'p'})

    kappa0 = df.name[df.index.names.index('kappa0')]
    p = belief['p'].copy()
    if kappa0 < 0:
        p = 1 - p
    belief.loc[:, 'p correct'] = p

    belief.name = df.name
    return belief


def run(results_path, seed):
    np.random.seed(seed)

    results = pd.read_csv(
        path(results_path).dirname().joinpath("model_belief_by_trial.csv"))
    results['p'] = np.exp(results['logp'])

    all_belief = []

    cols = ['model', 'likelihood', 'query', 'sigma', 'phi', 'kappa0']
    for key, df in results.groupby(cols):
        print key
        model, lh, query, sigma, phi, kappa0 = key
        belief = df\
            .set_index(
                ['version', 'pid', 'stimulus', 'trial', 'hypothesis'])['p']\
            .unstack('hypothesis')

        mask = np.zeros(len(belief.columns))
        mask[belief.columns > 0] = 1
        mask[belief.columns == 0] = 0.5

        belief = (belief * mask)\
            .sum(axis=1)\
            .reset_index()\
            .rename(columns={0: 'p'})

        p = belief['p'].copy()
        if kappa0 < 0:
            p = 1 - p
        belief.loc[:, 'p correct'] = p

        for col, val in zip(cols, key):
            belief[col] = val

        all_belief.append(belief)

    belief = pd\
        .concat(all_belief)\
        .set_index([
            'model', 'likelihood', 'query', 'version', 'kappa0', 'pid'])\
        .sortlevel()

    belief.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
