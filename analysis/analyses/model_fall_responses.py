#!/usr/bin/env python

"""
Computes the average model queries for "will it fall?". Produces a csv
file with the following columns:

    query (string)
        the model query
    block (string)
        the experiment phase
    kappa0 (float)
        the true log mass ratio
    stimulus (string)
        the name of the stimulus
    lower (float in [0, 1])
        lower bound of the 95% confidence interval
    median (float in [0, 1])
        median value across the bootstrapped means
    upper (float in [0, 1])
        upper bound of the 95% confidence interval

"""

import util
import pandas as pd
import numpy as np


def percent_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    return (samps / 10.0).apply(util.bootstrap_mean)


def more_than_half_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    return (samps > 5).apply(util.beta)


def more_than_one_fell(data):
    samps = data.pivot(
        index='sample',
        columns='kappa',
        values='nfell')
    return (samps > 1).apply(util.beta)


def compute_query(data, query):
    result = data\
        .groupby(level=['sigma', 'phi', 'stimulus'])\
        .apply(query)\
        .unstack()\
        .stack('kappa')\
        .reset_index()\
        .rename(columns={
            'kappa': 'kappa0'
        })
    result['query'] = query.__name__
    return result


def run(dest, seed):
    np.random.seed(seed)
    ipe = util.load_model()[0]
    results = []

    for block in ['A', 'B']:
        model = compute_query(ipe[block], percent_fell)
        model['block'] = block
        results.append(model)

        model = compute_query(ipe[block], more_than_half_fell)
        model['block'] = block
        results.append(model)

        model = compute_query(ipe[block], more_than_one_fell)
        model['block'] = block
        results.append(model)

    results = pd\
        .concat(results)\
        .set_index(['query', 'block', 'kappa0', 'stimulus'])\
        .sortlevel()

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(__doc__, seed=True)
    args = parser.parse_args()
    run(args.dest, args.seed)

