#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "model_belief_agg.csv"


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
        results_path.joinpath("model_belief.csv"))
    results['p'] = np.exp(results['logp'])

    all_belief = []

    for likelihood, df in results.groupby('likelihood'):
        cols = ['model', 'likelihood', 'version', 'kappa0']
        assert likelihood in ['ipe', 'empirical']
        if likelihood == 'ipe':
            cols.extend(['sigma', 'phi'])

        belief = df\
            .set_index(cols + ['pid', 'stimulus', 'trial', 'hypothesis'])\
            .groupby(level=cols)\
            .apply(agg_belief)\
            .reset_index(-1, drop=True)\
            .reset_index()

        if likelihood == 'empirical':
            belief['sigma'] = np.nan
            belief['phi'] = np.nan

        all_belief.append(belief)

    belief = pd\
        .concat(all_belief)\
        .set_index(['model', 'likelihood', 'version', 'kappa0',
                    'pid', 'stimulus', 'trial'])

    pth = results_path.joinpath(filename)
    belief.to_csv(pth)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
