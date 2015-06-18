#!/usr/bin/env python

"""
Computes the bayes factor for every participant under each
of the different models. Produces a csv file with the following columns:

    likelihood (string)
        the likelihood (e.g. empirical, ipe)
    counterfactual (bool)
        whether the counterfactual likelihood was used
    model (string)
        the model type (e.g. static, learning)
    version (string)
        the experiment version
    pid (string)
        the participant id
    K (float)
        the bayes factor

"""

__depends__ = ["marginal_likelihoods.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    model = pd\
        .read_csv(os.path.join(results_path, "marginal_likelihoods.csv"))\
        .set_index(['likelihood', 'counterfactual', 'version', 'num_mass_trials', 'pid', 'model'])['logp']\
        .unstack('model')

    results = (model['learning'] - model['static']).to_frame('logK')
    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
