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

def fit(df):
    data = df.set_index(['kappa0', 'stimulus'])
    human = data['human'] ** 2
    model = data['model'] ** 2

    results = []
    for k in range(1, 7):

        # compute the slope
        slope = 1.0 / k

        # fit the intercept
        X = np.ones((model.size, 1))
        Y = np.asarray(human - slope * model)[:, None]
        intercept = float(np.linalg.lstsq(X, Y)[0].ravel())

        # compute MSE
        X = slope * model + intercept
        Y = human
        err = np.mean((Y - X) ** 2)

        results.append({
            'k': k,
            'slope': slope,
            'intercept': intercept,
            'mse': err
        })

    results = pd.DataFrame(results).set_index('k')
    results.name = df.name
    return results


def run(dest, results_path):
    human = pd.read_csv(os.path.join(results_path, "human_fall_responses.csv"))
    human = human[['version', 'block', 'kappa0', 'stimulus', 'stddev']]\
        .rename(columns={'stddev': 'human'})

    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))
    model = model[['query', 'block', 'kappa0', 'stimulus', 'stddev']]\
        .rename(columns={'stddev': 'model'})

    results = pd.merge(human, model)\
        .groupby(['query', 'version', 'block'])\
        .apply(fit)

    results.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
