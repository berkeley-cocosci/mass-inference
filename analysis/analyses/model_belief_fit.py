#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    query = util.get_query()
    belief = pd.read_csv(path(results_path).dirname().joinpath(
        "model_belief_agg.csv"))
    belief = belief.set_index('query').ix[query]

    empirical = belief\
        .groupby('likelihood')\
        .get_group('empirical')\
        .set_index(['model', 'likelihood', 'version', 'pid',
                    'kappa0', 'stimulus', 'trial'])\
        .copy()

    ipe = belief\
        .groupby('likelihood')\
        .get_group('ipe')\
        .set_index(['model', 'likelihood', 'version', 'pid',
                    'kappa0', 'stimulus', 'trial'])\
        .copy()

    results = pd\
        .concat([empirical, ipe])\
        .reset_index()\
        .set_index(['model', 'likelihood', 'version', 'trial'])\
        .sortlevel()

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
