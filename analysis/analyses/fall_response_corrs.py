#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import util
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    pth = path(results_path).dirname().joinpath("fall_responses.csv")
    means = pd\
        .read_csv(pth)\
        .set_index(['version', 'block', 'species',
                    'kappa0', 'stimulus'])['median']\
        .groupby(level='version')\
        .get_group('GH')

    results = {}
    for block, df in means.groupby(level='block'):
        m = df.unstack(['species', 'kappa0'])

        # human vs human
        x = m[('human', -1.0)]
        y = m[('human', 1.0)]
        results[(block, 'Human', 'Human')] = util.bootcorr(x, y)

        # mass-sensitive ipe vs human
        x = pd.concat([m[('model', -1.0)], m[('model', 1.0)]])
        y = pd.concat([m[('human', -1.0)], m[('human', 1.0)]])
        results[(block, 'ModelS', 'Human')] = util.bootcorr(x, y)

        # mass-insensitive ipe vs human
        x = pd.concat([m[('model', 0.0)], m[('model', 0.0)]])
        y = pd.concat([m[('human', -1.0)], m[('human', 1.0)]])
        results[(block, 'ModelIS', 'Human')] = util.bootcorr(x, y)

    results = pd.DataFrame.from_dict(results).T
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['block', 'X', 'Y'])

    results.to_csv(results_path)


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
