#!/usr/bin/env python

import util
import pandas as pd
import numpy as np

filename = "mass_accuracy_by_stimulus.csv"


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()
    results = []

    model_belief = pd.read_csv(
        results_path.joinpath('model_belief_agg.csv'))

    correct = human['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .groupby(['version', 'kappa0', 'stimulus'])['mass? correct']\
        .apply(util.beta)\
        .unstack(-1)\
        .reset_index()
    correct['class'] = 'static'
    correct['species'] = 'human'
    correct = correct\
        .set_index(['species', 'class', 'version', 'kappa0', 'stimulus'])\
        .stack()
    results.append(correct)

    belief = model_belief\
        .groupby('model')\
        .get_group('static')\
        .groupby(['likelihood', 'version', 'kappa0', 'stimulus'])['p correct']\
        .apply(util.bootstrap_mean)\
        .unstack(-1)\
        .reset_index()\
        .rename(columns={'likelihood': 'species'})
    belief['class'] = 'static'
    belief = belief\
        .set_index(['species', 'class', 'version', 'kappa0', 'stimulus'])\
        .stack()
    results.append(belief)

    results = pd.concat(results).unstack().sortlevel()

    pth = results_path.joinpath(filename)
    results.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
