#!/usr/bin/env python

"""
Computes the overall log likelihood ratio between the learning model and the
static model. Produces a csv with the following columns:

    likelihood (string)
        the name of the likelihood
    counterfactual (bool)
        whether the counterfactual likelihood was used
    fitted (bool)
        whether the model was fitted to human data
    version (string)
        the experiment version
    num_mass_trials (int)
        the number of mass trials on which participants responded
    llhr (float)
        the log likelihood ratio of learning to static
    model_favored (string)
        the name of the model that is favored by the LLHR
"""

__depends__ = ["model_log_lh_ratio_by_trial.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):

    llhr = pd\
        .read_csv(os.path.join(results_path, 'model_log_lh_ratio_by_trial.csv'))\
        .groupby(['likelihood', 'counterfactual', 'fitted', 'version', 'num_mass_trials'])['llhr']\
        .sum()\
        .to_frame('llhr')

    llhr['model_favored'] = 'equal'
    llhr.loc[llhr['llhr'] < 0, 'model_favored'] = 'static'
    llhr.loc[llhr['llhr'] > 0, 'model_favored'] = 'learning'

    llhr.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
