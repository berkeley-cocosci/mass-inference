#!/usr/bin/env python

"""
Computes bootstrapped correlations for responses to "will it fall?" between:

    * human vs. human
    * mass-sensitive ipe vs. human
    * mass-insensitive ipe vs. human

Outputs a csv file with the following columns:

    block (string)
        experiment block/phase
    X (string)
        name of the x-variable (Human, ModelS, ModelIS)
    Y (string)
        name of the y-variable (Human, ModelS, ModelIS)
    lower (float)
        lower bound of the 95% confidence interval
    median
        median of the bootstrapped correlations
    upper
        upper bound of the 95% confidence interval

"""

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
    parser = util.default_argparser(__doc__)
    args = parser.parse_args()
    run(args.results_path, args.seed)
