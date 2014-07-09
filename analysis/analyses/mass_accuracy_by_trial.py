#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_accuracy_by_trial.csv"


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()
    results = []

    model_belief = pd.read_csv(
        results_path.joinpath('model_belief_agg.csv'))

    def choose_first(x):
        version, num, pid = x.name
        if version == 'I' and num == -1:
            ix = x['mass? response'].dropna().index[0]
            y = x.copy()
            y['mass? response'] = np.nan
            y.loc[ix, 'mass? response'] = x['mass? response'].ix[ix]
        else:
            y = x.copy()
        return y

    responses = human['C']\
        .set_index(['version', 'num_mass_trials', 'pid', 'trial'])\
        .groupby(level=['version', 'num_mass_trials', 'pid'])\
        .apply(choose_first)\
        .reset_index()\
        .dropna(axis=0, subset=['mass? response'])

    for model in list(model_belief['model'].unique()):
        for kappa in [-1.0, 1.0, 'all']:
            if kappa == 'all':
                correct = responses
                belief = model_belief
            else:
                correct = responses.groupby('kappa0').get_group(kappa)
                belief = model_belief.groupby('kappa0').get_group(kappa)

            correct = correct\
                .groupby(['version', 'num_mass_trials',
                          'trial'])['mass? correct']\
                .apply(util.beta)\
                .unstack(-1)\
                .reset_index()
            correct['class'] = model
            correct['species'] = 'human'
            correct['kappa0'] = kappa
            correct = correct\
                .set_index(['species', 'class', 'version', 'kappa0',
                            'num_mass_trials', 'trial'])\
                .stack()
            results.append(correct)

            belief = belief\
                .groupby('model')\
                .get_group(model)\
                .groupby(['likelihood', 'version', 'trial'])['p correct']\
                .apply(util.bootstrap_mean)\
                .unstack(-1)\
                .reset_index()\
                .rename(columns={'likelihood': 'species'})
            belief['class'] = model
            belief['kappa0'] = kappa
            belief['num_mass_trials'] = -1
            belief = belief\
                .set_index(['species', 'class', 'version', 'kappa0', 'num_mass_trials', 'trial'])\
                .stack()
            results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
