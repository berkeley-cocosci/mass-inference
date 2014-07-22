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
        'model_belief_fit.csv'))

    for kappa in [-1.0, 1.0, 'all']:
        if kappa == 'all':
            correct = human['C']
            belief = model_belief
        else:
            correct = human['C'].groupby('kappa0').get_group(kappa)
            belief = model_belief.groupby('kappa0').get_group(kappa)

        correct = correct\
            .dropna(axis=0, subset=['mass? response'])\
            .groupby('version')['mass? correct']\
            .apply(util.beta)\
            .unstack(-1)\
            .reset_index()
        correct['class'] = 'static'
        correct['species'] = 'human'
        correct['kappa0'] = kappa
        correct = correct\
            .set_index(['species', 'class', 'version', 'kappa0'])\
            .stack()
        results.append(correct)

        belief = belief\
            .groupby('model')\
            .get_group('static')\
            .groupby(['likelihood', 'version'])['p correct']\
            .median()
        belief.name = 'median'
        belief = belief\
            .reset_index()\
            .rename(columns={'likelihood': 'species'})
        belief['class'] = 'static'
        belief['kappa0'] = kappa
        belief = belief\
            .set_index(['species', 'class', 'version', 'kappa0'])\
            .stack()
        results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
