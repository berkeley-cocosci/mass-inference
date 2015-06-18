#!/usr/bin/env python

"""
Computes the bayes factor for each model. Produces a csv file with the following
columns:

    likelihood (string)
        the likelihood (e.g. empirical, ipe)
    counterfactual (bool)
        whether the counterfactual likelihood was used
    model (string)
        the model type (e.g. static, learning)
    version (string)
        the experiment version
    K (float)
        the bayes factor

"""

__depends__ = ["bayes_factors_by_participant.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd\
        .read_csv(os.path.join(results_path, "bayes_factors_by_participant.csv"))\
        .groupby(['likelihood', 'counterfactual', 'version'])['logK']\
        .sum()\
        .to_frame('logK')

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
