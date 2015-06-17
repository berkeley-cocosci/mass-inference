#!/usr/bin/env python

"""
Computes the spearman rank corrlation between trial number and mass accuracy for
each experiment. This produces a csv file with the following columns:

    version (string)
        the experiment version
    num_mass_trials (string)
        the number of trials on which people responded. -1 indicates a
        between-subjects analysis
    lower (float)
        lower bound of the 95% confidence interval
    median
        median of the bootstrap distribution
    upper
        upper bound of the 95% confidence interval

"""

__depends__ = ["human_mass_accuracy_by_trial.csv"]
__random__ = True
__parallel__ = True

import os
import util
import pandas as pd
import numpy as np
import scipy.stats

from IPython.parallel import Client, require


@require('numpy as np', 'pandas as pd')
def corr(df):
    name, df = df
    x = np.asarray(df['trial'])
    y = np.asarray(df['median'])
    N = np.asarray(df['N'])
    if y.size == 1:
        stats = pd.Series(
            [np.nan, np.nan, np.nan],
            index=['lower', 'median', 'upper'])
    else:
        f = lambda i: (np.random.rand(10000, N[i]) < y[i]).mean(axis=1)
        samps = np.array([f(i) for i in range(y.size)]).T
        corrs = np.array([scipy.stats.spearmanr(x, s)[0] for s in samps])
        stats = pd.Series(
            np.percentile(corrs, [2.5, 50, 97.5]),
            index=['lower', 'median', 'upper'])
    stats.name = name
    return stats


def run(dest, results_path, seed, parallel):
    np.random.seed(seed)

    # load in human data
    human = pd\
        .read_csv(os.path.join(results_path, "human_mass_accuracy_by_trial.csv"))\
        .groupby('kappa0')\
        .get_group('all')

    def as_df(x, index_names):
        df = pd.DataFrame(x)
        if len(index_names) == 1:
            df.index.name = index_names[0]
        else:
            df.index = pd.MultiIndex.from_tuples(df.index)
            df.index.names = index_names
        return df

    if parallel:
        rc = Client()
        dview = rc[:]
        mapfunc = dview.map_sync
    else:
        mapfunc = map

    # compute correlations
    results = mapfunc(corr, list(human.groupby(['version', 'num_mass_trials'])))
    results = as_df(results, ['version', 'num_mass_trials'])
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed, args.parallel)

