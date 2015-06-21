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

__depends__ = ["human_fall_responses.csv", "human_fall_responses_raw.csv", "single_model_fall_responses.csv"]
__random__ = True
__parallel__ = True

import pandas as pd
import numpy as np
import util
import os
import scipy.stats


def corr(arr):
    ix = np.arange(arr.shape[1])
    np.random.shuffle(ix)
    half1 = np.nanmean(arr[:, ix[:len(ix)/2]], axis=1)
    half2 = np.nanmean(arr[:, ix[len(ix)/2:]], axis=1)
    return scipy.stats.pearsonr(half1, half2)[0]


def run(dest, results_path, seed, version):
    np.random.seed(seed)

    human = pd\
        .read_csv(os.path.join(results_path, "human_fall_responses.csv"))\
        .set_index(['version', 'block', 'stimulus', 'kappa0'])['median']\
        .sortlevel()

    human_raw = pd\
        .read_csv(os.path.join(results_path, "human_fall_responses_raw.csv"))\
        .set_index(['version', 'block', 'stimulus', 'kappa0', 'pid'])['fall? response']\
        .sortlevel()

    human_version = human.ix[version]
    human_exp1 = human.ix['H']
    human_exp2 = human.ix['G']

    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))

    results = {}
    for block in ['A', 'B']:
        h1 = human_exp1.ix[block]
        h2 = human_exp2.ix[block]
        results[(block, 'H', 'G')] = util.bootcorr(h1, h2)

        for query in model['query'].unique():
            hv = human_version.ix[block]
            m = model\
                .groupby(['block', 'query'])\
                .get_group((block, query))\
                .set_index(['stimulus', 'kappa0'])['median']\
                .sortlevel()\
                .ix[hv.index]

            results[(block, query, 'Human')] = util.bootcorr(m, hv)

        for version in ['H', 'G']:
            hraw = np.asarray(human_raw.ix[(version, block)].unstack('pid'))
            corrs = np.array([corr(hraw) for _ in range(10000)])
            results[(block, version, version)] = pd.Series(
                np.percentile(corrs, [2.5, 50, 97.5]),
                index=['lower', 'median', 'upper'])

    results = pd.DataFrame.from_dict(results).T
    results.index = pd.MultiIndex.from_tuples(
        results.index,
        names=['block', 'X', 'Y'])

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
