#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path
from util import exponentiated_luce_choice as elc


def run(results_path, seed):
    np.random.seed(seed)

    query = util.get_query()
    belief = pd.read_csv(path(results_path).dirname().joinpath(
        "model_belief_agg.csv"))
    belief = belief.set_index('query').ix[query]

    p = pd.read_csv(path(results_path).dirname().joinpath(
        "fit_mass_responses.csv")).set_index('model')['median']

    empirical = belief\
        .groupby('likelihood')\
        .get_group('empirical')\
        .set_index(['model', 'likelihood', 'version', 'pid',
                    'kappa0', 'stimulus', 'trial'])\
        .copy()
    empirical.loc[:, 'p'] = elc(empirical['p'], p['empirical'])
    empirical.loc[:, 'p correct'] = elc(
        empirical['p correct'], p['empirical'])

    ipe = belief\
        .groupby('likelihood')\
        .get_group('ipe')\
        .set_index(['model', 'likelihood', 'version', 'pid',
                    'kappa0', 'stimulus', 'trial'])\
        .copy()
    ipe.loc[:, 'p'] = elc(ipe['p'], p['ipe'])
    ipe.loc[:, 'p correct'] = elc(ipe['p correct'], p['ipe'])

    results = pd\
        .concat([empirical, ipe])\
        .reset_index()\
        .set_index(['model', 'likelihood', 'version', 'trial'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
