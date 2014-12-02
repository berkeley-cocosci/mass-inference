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

__depends__ = ["human_mass_responses_by_stimulus.csv", "model_mass_responses_by_stimulus.csv"]
__random__ = True

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path, seed):
    np.random.seed(seed)

    # load in human data
    human = pd\
        .read_csv(os.path.join(results_path, "human_mass_responses_by_stimulus.csv"))\
        .set_index(['stimulus', 'kappa0', 'version'])['median']\
        .unstack('version')

    # load in model data
    cols = ['likelihood', 'counterfactual', 'model', 'fitted']
    model = pd\
        .read_csv(os.path.join(results_path, "model_mass_responses_by_stimulus.csv"))\
        .set_index(cols + ['stimulus', 'kappa0', 'version'])['median']\
        .unstack('version')

    # create empty dataframe for our results
    results = pd.DataFrame([])

    for version in human:
        corr = model[version]\
            .groupby(level=cols)\
            .apply(util.bootcorr, human[version], method='pearson')\
            .unstack()\
            .reset_index()

        corr['version'] = version
        results = results.append(corr)

    results = results.set_index(['version'] + cols).sortlevel()
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed)
