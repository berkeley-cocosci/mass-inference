#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path
from util import exponentiated_luce_choice as elc


def run(results_path, seed):
    np.random.seed(seed)

    belief = pd.read_csv(path(results_path).dirname().joinpath(
        "model_belief_agg.csv"))

    p = pd.read_csv(path(results_path).dirname().joinpath(
        "fit_mass_responses.csv")).set_index('model')['median']

    empirical = belief\
        .groupby('likelihood')\
        .get_group('empirical')\
        .copy()
    empirical.loc[:, 'p'] = elc(empirical['p'], p['empirical'])
    empirical.loc[:, 'p correct'] = elc(
        empirical['p correct'], p['empirical'])

    ipe = belief\
        .groupby('likelihood')\
        .get_group('ipe')\
        .copy()
    ipe.loc[:, 'p'] = elc(ipe['p'], p['ipe'])
    ipe.loc[:, 'p correct'] = elc(ipe['p correct'], p['ipe'])

    results = pd\
        .concat([empirical, ipe])\
        .set_index(['model', 'likelihood', 'version', 'trial'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
