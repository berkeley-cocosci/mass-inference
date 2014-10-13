#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()
    results = []

    model_belief = pd.read_csv(path(results_path).dirname().joinpath(
        'model_belief.csv'))

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

    human_I = human['C'].groupby('version').get_group('I').copy()
    human_I['num_mass_trials'] = -1
    human_I = human_I\
        .set_index(['version', 'num_mass_trials', 'pid', 'trial'])\
        .groupby(level=['version', 'num_mass_trials', 'pid'])\
        .apply(choose_first)\
        .reset_index()

    model_belief = model_belief\
        .set_index(['pid', 'stimulus', 'trial'])\
        .drop('prior', level='stimulus')
    for pid, df in human['C'].groupby('pid'):
        num_mass_trials = int(df['num_mass_trials'].unique())
        model_belief.loc[pid, 'num_mass_trials'] = num_mass_trials
        responses = df.set_index(
            ['pid', 'stimulus', 'trial'])['mass? response']
        ix = responses[responses.isnull()].index
        model_belief.drop(ix, inplace=True)
    model_belief = model_belief.reset_index()

    belief_I = model_belief\
        .groupby('version')\
        .get_group('I')\
        .copy()\
        .set_index(['pid', 'stimulus', 'trial'])\
        .sortlevel()

    for pid, df in human_I.groupby('pid'):
        responses = df.set_index(
            ['pid', 'stimulus', 'trial'])['mass? response']
        ix = responses[responses.isnull()].index
        belief_I.drop(ix, inplace=True)

    belief_I = belief_I.reset_index()
    belief_I['num_mass_trials'] = -1

    responses = pd\
        .concat([human['C'], human_I])\
        .dropna(subset=['mass? response'], axis=0)
    model_belief = pd.concat([model_belief, belief_I])

    for model in list(model_belief['model'].unique()):
        for kappa in [-1.0, 1.0, 'all']:
            if kappa == 'all':
                correct = responses
                belief = model_belief
            else:
                correct = responses.groupby('kappa0').get_group(kappa)
                belief = model_belief.groupby('kappa0').get_group(kappa)

            if model == 'chance':
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
                .groupby(['likelihood', 'version', 'num_mass_trials',
                          'trial'])['p correct']\
                .mean()
            belief.name = 'median'
            belief = belief\
                .reset_index()\
                .rename(columns={'likelihood': 'species'})
            belief['class'] = model
            belief['kappa0'] = kappa
            belief = belief\
                .set_index(['species', 'class', 'version', 'kappa0',
                            'num_mass_trials', 'trial'])\
                .stack()
            results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
