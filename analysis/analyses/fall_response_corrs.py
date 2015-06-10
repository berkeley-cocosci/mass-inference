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

__depends__ = ["human_fall_responses.csv", "single_model_fall_responses.csv"]
__random__ = True

import pandas as pd
import numpy as np
import util
import os


def run(dest, results_path, seed, version):
    np.random.seed(seed)

    human = pd.read_csv(os.path.join(results_path, "human_fall_responses.csv"))
    human = human.groupby('version').get_group(version)

    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))

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
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to use responses from')
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed, args.version)
