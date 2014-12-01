#!/usr/bin/env python

"""
Computes the log likelihood ratio between learning and static models for each
participant. Produces a csv file with the following columns:

    likelihood (string)
        the name of the likelihood
    counterfactual (bool)
        whether the counterfactual likelihood was used
    fitted (bool)
        whether the model was fitted to human data
    version (string)
        the experiment version
    num_mass_trials (int)
        the number of mass trials that participant responded on
    pid (string)
        the participant's unique id
    llhr (float)
        the log likelihood ratio of learning to static
    model_favored (string)
        the name of the model that is favored by the LLHR

"""

__depends__ = ["model_log_lh.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):

    data = pd.read_csv(os.path.join(results_path, 'model_log_lh.csv'))

    cols = ['likelihood', 'counterfactual', 'model', 'fitted', 'version', 'num_mass_trials', 'pid']
    llh = data\
        .groupby(cols)['llh']\
        .sum()\
        .unstack('model')

    llhr = (llh['learning'] - llh['static']).to_frame('llhr')
    llhr['model_favored'] = 'equal'
    llhr.loc[llhr['llhr'] < 0, 'model_favored'] = 'static'
    llhr.loc[llhr['llhr'] > 0, 'model_favored'] = 'learning'

    llhr.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
