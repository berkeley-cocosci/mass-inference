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

    def choose_first(x):
        version, pid = x.name
        if version == 'I':
            ix = x['mass? response'].dropna().index[0]
            y = x.copy()
            y['mass? response'] = np.nan
            y.loc[ix, 'mass? response'] = x['mass? response'].ix[ix]
        else:
            y = x.copy()
        return y

    responses = data['human']['C']\
        .set_index(['version', 'pid', 'trial'])\
        .groupby(level=['version', 'pid'])\
        .apply(choose_first)\
        .reset_index()\
        .dropna(axis=0, subset=['mass? response'])

    for model in list(model_belief['model'].unique()):
        human = responses\
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
