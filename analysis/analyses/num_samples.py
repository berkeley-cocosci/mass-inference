#!/usr/bin/env python

"""
Analyzes the variance of model responses and of human responses to estimate the
number of samples that participants take for each response. Saves out a CSV with
the following columns:


"""

__depends__ = ["human_fall_responses.csv", "single_model_fall_responses.csv"]

import os
import util
import pandas as pd
import numpy as np


def run(dest, results_path, version, block):
    human = pd.read_csv(os.path.join(results_path, "human_fall_responses.csv"))
    human = human\
        .groupby(['version', 'block'])\
        .get_group((version, block))\
        .set_index(['kappa0', 'stimulus'])\
        .sortlevel()

    query = util.get_query()
    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))
    model = model\
        .groupby(['query', 'block'])\
        .get_group((query, block))\
        .set_index(['kappa0', 'stimulus'])\
        .sortlevel()
    model = model.ix[human.index]

    results = []
    for k in range(1, 7):

        # compute the slope
        slope = 1.0 / k

        # fit the intercept
        X = np.ones((model['stddev'].size, 1))
        Y = np.asarray(human['stddev']**2 - slope * model['stddev']**2)[:, None]
        intercept = float(np.linalg.lstsq(X, Y)[0].ravel())

        # compute MSE
        X = slope * model['stddev']**2 + intercept
        Y = human['stddev']**2
        err = np.mean((Y - X) ** 2)

        results.append({
            'k': k,
            'slope': slope,
            'intercept': intercept,
            'mse': err
        })

    results = pd.DataFrame(results).set_index('k')
    results.to_csv(dest)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to use responses from')
    parser.add_argument(
        '--block',
        default='B',
        help='which block of the experiment to use responses from')
    args = parser.parse_args()
    run(args.to, args.results_path, args.version, args.block)
