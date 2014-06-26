#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_accuracy.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    results = []

    model_belief = pd.read_csv(
        results_path.joinpath('model_belief_agg.csv'))

    for kappa in [-1.0, 1.0, 'all']:
        if kappa == 'all':
            human = data['human']['C']
            belief = model_belief
        else:
            human = data['human']['C'].groupby('kappa0').get_group(kappa)
            belief = model_belief.groupby('kappa0').get_group(kappa)

        human = human\
            .dropna(axis=0, subset=['mass? response'])\
            .groupby('version')['mass? correct']\
            .apply(util.beta)\
            .unstack(-1)\
            .reset_index()
        human['class'] = 'static'
        human['species'] = 'human'
        human['kappa0'] = kappa
        human = human\
            .set_index(['species', 'class', 'version', 'kappa0'])\
            .stack()
        results.append(human)

        belief = belief\
            .groupby('model')\
            .get_group('static')\
            .groupby(['likelihood', 'version'])['p correct']\
            .apply(util.bootstrap_mean)\
            .unstack(-1)\
            .reset_index()\
            .rename(columns={'likelihood': 'species'})
        belief['class'] = 'static'
        belief['kappa0'] = kappa
        belief = belief\
            .set_index(['species', 'class', 'version', 'kappa0'])\
            .stack()
        results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
