#!/usr/bin/env python

"""
Computes the pearson correlation between model and human judgments on "which is
heavier?" for each version of the experiment. Produces a csv file with the
following columns:

    version (string)
        the experiment version
    likelihood (string)
        the name of the likelihood that was used
    counterfactual (bool)
        whether the counterfactual likelihood was used
    model (string)
        the name of the model (e.g. learning, static)
    fitted (bool)
        whether the model was fitted to human data
    lower (float)
        lower bound of the 95% confidence interval
    median (float)
        median of the bootstrap distribution
    upper (float)
        upper bound of the 95% confidence interval

"""

__depends__ = ["human", "single_model_belief.csv"]
__random__ = True

import os
import util
import pandas as pd
import numpy as np
import scipy.stats


def run(dest, results_path, data_path, seed):
    np.random.seed(seed)

    def bootcorr(df, n=10000):
        print df.name

        by_pid = df.unstack('pid')
        h = np.asarray(by_pid['h'])
        m = np.asarray(by_pid['m'])
        num_pids = h.shape[1]

        ix = np.random.randint(0, num_pids, (num_pids, n))
        corrs = np.empty(n)
        for i in range(n):
            h_mean = np.nanmean(h[:, ix[:, i]], axis=1)
            m_mean = np.nanmean(m[:, ix[:, i]], axis=1)
            corrs[i] = scipy.stats.pearsonr(h_mean, m_mean)[0]

        stats = pd.Series(
            np.percentile(corrs[~np.isnan(corrs)], [2.5, 50, 97.5]),
            index=['lower', 'median', 'upper'])

        return stats

    # load in human data
    human = util.load_human(data_path)['C']\
        .dropna(axis=0, subset=['mass? response'])

    # convert from -1, 1 responses to 0, 1 responses
    human.loc[:, 'mass? response'] = (human['mass? response'] + 1) / 2.0
    human = human[['stimulus', 'kappa0', 'version', 'pid', 'mass? response']]
    human = human.rename(columns={'mass? response': 'h'})

    # load in model data
    model = pd.read_csv(os.path.join(results_path, 'single_model_belief.csv'))
    cols = ['version', 'likelihood', 'counterfactual', 'model', 'fitted']
    model = model[cols + ['stimulus', 'kappa0', 'pid', 'p']]
    model = model.rename(columns={'p': 'm'})

    # merge the data
    data = pd\
        .merge(human, model)\
        .set_index(cols + ['stimulus', 'kappa0', 'pid'])\
        .sortlevel()

    results = data\
        .groupby(level=cols)\
        .apply(bootcorr)

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.seed)
