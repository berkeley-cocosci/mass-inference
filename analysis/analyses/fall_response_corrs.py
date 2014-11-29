#!/usr/bin/env python

"""
Computes bootstrapped correlations for responses to "will it fall?" between:

    * human vs. human
    * mass-sensitive ipe vs. human
    * mass-insensitive ipe vs. human

Outputs a csv file with the following columns:

    query (string)
        the model query
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
import os


def run(dest, results_path, seed):
    np.random.seed(seed)

    human = pd.read_csv(os.path.join(results_path, "human_fall_responses.csv"))
    human = human.groupby('version').get_group('GH')

    model = pd.read_csv(os.path.join(results_path, "model_fall_responses.csv"))

    results = {}
    for (query, block), df in model.groupby(['query', 'block']):
        h = human\
            .groupby('block')\
            .get_group(block)\
            .pivot('stimulus', 'kappa0', 'median')

        m = df.pivot('stimulus', 'kappa0', 'median')

        # human vs human
        x = h[-1.0]
        y = h[1.0]
        results[(query, block, 'Human', 'Human')] = util.bootcorr(x, y)

        # mass-sensitive ipe vs human
        x = pd.concat([m[-1.0], m[1.0]])
        y = pd.concat([h[-1.0], h[1.0]])
        results[(query, block, 'ModelS', 'Human')] = util.bootcorr(x, y)

        # mass-insensitive ipe vs human
        x = pd.concat([m[0.0], m[0.0]])
        y = pd.concat([h[-1.0], h[1.0]])
        results[(query, block, 'ModelIS', 'Human')] = util.bootcorr(x, y)

    results = pd.DataFrame.from_dict(results).T
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['query', 'block', 'X', 'Y'])

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(__doc__, results_path=True, seed=True)
    args = parser.parse_args()
    run(args.dest, args.results_path, args.seed)
