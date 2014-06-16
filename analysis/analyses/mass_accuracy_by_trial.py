#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_accuracy_by_trial.csv"


def run(data, results_path, seed):
    np.random.seed(seed)
    results = []

    model_belief = pd.read_csv(
        results_path.joinpath('model_belief_agg.csv'))

    human = data['human']['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .set_index(['version', 'pid', 'trial'])['mass? correct']
    #bad = ~(human.unstack('trial')[20]).astype('bool')
    #bad = bad.index[np.asarray(bad)]
    bad = []

    for model in list(model_belief['model'].unique()):
        human = data['human']['C']\
            .dropna(axis=0, subset=['mass? response'])\
            .set_index(['version', 'pid'])\
            .drop(bad)\
            .reset_index()\
            .groupby(['version', 'kappa0', 'trial'])['mass? correct']\
            .apply(util.beta)\
            .unstack(-1)\
            .reset_index()
        human['class'] = model
        human['species'] = 'human'
        human = human\
            .set_index(['species', 'class', 'version', 'kappa0', 'trial'])\
            .stack()
        results.append(human)

        belief = model_belief\
            .groupby('model')\
            .get_group(model)\
            .set_index(['version', 'pid'])\
            .drop(bad)\
            .reset_index()\
            .groupby(['likelihood', 'version', 'kappa0', 'trial'])['p correct']\
            .apply(util.bootstrap_mean)\
            .unstack(-1)\
            .reset_index()\
            .rename(columns={'likelihood': 'species'})
        belief['class'] = model
        belief = belief\
            .set_index(['species', 'class', 'version', 'kappa0', 'trial'])\
            .stack()
        results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
