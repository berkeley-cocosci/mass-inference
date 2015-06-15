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

__depends__ = ["human_fall_responses_raw.csv", "single_model_fall_responses_raw.csv"]
__random__ = True

import pandas as pd
import numpy as np
import util
import os
import scipy.stats


def bootsamps(x, func=np.mean, n=10000):
    x_ = x.dropna()
    ix = np.random.randint(0, x_.size, (n, x_.size))
    samps = func(np.asarray(x_)[ix], axis=1)
    samps = pd.Series(samps, name=x.name)
    samps.index.name = 'sample'
    return samps


def corr(x, y):
    df = pd.DataFrame(dict(x=x, y=y))
    corrs = df.groupby(level='sample').apply(lambda x: scipy.stats.pearsonr(x['x'], x['y'])[0])
    r = pd.Series(np.percentile(corrs, [2.5, 50, 97.5]), index=['lower', 'median', 'upper'])
    return r


def run(dest, results_path, seed, version):
    np.random.seed(seed)

    human = pd.read_csv(os.path.join(results_path, "human_fall_responses_raw.csv"))
    human = human.groupby('version').get_group(version)

    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses_raw.csv"))

    results = {}
    for (query, block), df in model.groupby(['query', 'block']):
        print query, block

        h = human\
            .groupby('block')\
            .get_group(block)\
            .groupby(['stimulus', 'kappa0'])['fall? response']\
            .apply(bootsamps)\
            .unstack('kappa0')

        m = df\
            .groupby(['stimulus', 'kappa0'])['response']\
            .apply(bootsamps)\
            .unstack('kappa0')

        # human vs human
        x = h[-1.0]
        y = h[1.0]
        results[(query, block, 'Human', 'Human')] = corr(x, y)

        # mass-sensitive ipe vs human
        x = pd.concat([m[-1.0], m[1.0]])
        y = pd.concat([h[-1.0], h[1.0]])
        results[(query, block, 'ModelS', 'Human')] = corr(x, y)

        # mass-insensitive ipe vs human
        x = pd.concat([m[0.0], m[0.0]])
        y = pd.concat([h[-1.0], h[1.0]])
        results[(query, block, 'ModelIS', 'Human')] = corr(x, y)

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
