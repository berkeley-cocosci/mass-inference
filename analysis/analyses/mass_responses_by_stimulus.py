#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)
    human = util.load_human()

    model_belief = pd.read_csv(path(results_path).dirname().joinpath(
        'model_belief_agg.csv'))

    human_C = human['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .copy()
    human_C.loc[:, 'mass? response'] = (human_C['mass? response'] + 1) / 2.0

    correct = human_C\
        .groupby(['version', 'kappa0', 'stimulus'])['mass? response']\
        .apply(util.beta)\
        .unstack(-1)\
        .reset_index()
    correct['class'] = 'static'
    correct['species'] = 'human'
    correct = correct\
        .set_index(['species', 'class', 'version', 'kappa0', 'stimulus'])\
        .stack()

    belief = model_belief\
        .groupby('model')\
        .get_group('static')\
        .groupby(['likelihood', 'version', 'kappa0', 'stimulus'])['p']\
        .mean()

    belief.name = 'median'
    belief = belief\
        .reset_index()\
        .rename(columns={'likelihood': 'species'})
    belief['class'] = 'static'
    belief = belief\
        .set_index(['species', 'class', 'version', 'kappa0', 'stimulus'])\
        .stack()

    results = pd\
        .concat([correct, belief])\
        .unstack()\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
