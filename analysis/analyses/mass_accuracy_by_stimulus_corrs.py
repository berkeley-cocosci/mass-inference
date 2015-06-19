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

__depends__ = ["human_mass_accuracy_by_stimulus.csv", "model_mass_accuracy_by_stimulus.csv"]
__random__ = True
__parallel__ = True

import os
import util
import pandas as pd
import numpy as np

from IPython.parallel import require


def run(dest, results_path, seed, parallel):
    np.random.seed(seed)

    # load in human data
    human = pd\
        .read_csv(os.path.join(results_path, "human_mass_accuracy_by_stimulus.csv"))\
        .set_index(['stimulus', 'kappa0', 'version'])['median']\
        .unstack('version')

    # load in model data
    cols = ['likelihood', 'counterfactual']
    model = pd\
        .read_csv(os.path.join(results_path, "model_mass_accuracy_by_stimulus.csv"))\
        .set_index(cols + ['stimulus', 'kappa0'])['median']

    @require('numpy as np', 'pandas as pd', 'util')
    def bootcorr(x, y, **kwargs):
        name, x = x
        corr = util.bootcorr(x, y, **kwargs)
        corr.name = name
        return corr

    # create empty dataframe for our results
    results = pd.DataFrame([])

    mapfunc = util.get_mapfunc(parallel)
    for version in human:
        corr = mapfunc(
            lambda df: bootcorr(df, human[version], method='pearson'),
            list(model.groupby(level=cols)))
        corr = util.as_df(corr, cols).reset_index()
        corr['version'] = version
        results = results.append(corr)

    results = results.set_index(['version'] + cols).sortlevel()
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed, args.parallel)
