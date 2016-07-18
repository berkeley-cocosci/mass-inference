#!/usr/bin/env python

__depends__ = [
    "human_mass_responses_by_stimulus.csv",
    "model_mass_responses_by_stimulus.csv",
]
__random__ = True

import util
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.optimize as optim


def sigmoid(x, b, offset=0.5, scale=1.0):
    return scale / (1.0 + np.exp(-(b * (x - offset))))


def fit_sigmoid(xarr, yarr):
    def fit(b):
        pred = sigmoid(xarr, b)
        diff = np.mean((yarr - pred) ** 2)
        return diff

    res = optim.minimize_scalar(fit)
    return res['x']


def bootfit(xarr, yarr, nboot=10000):
    x = np.asarray(xarr)
    y = np.asarray(yarr)
    idx = np.random.randint(0, len(x), (nboot, len(x)))
    b = np.array([fit_sigmoid(x[i], y[i]) for i in idx])
    lower, median, upper = np.percentile(b, [2.5, 50, 97.5])
    return pd.Series({'lower': lower, 'median': median, 'upper': upper})


def bootfit_random(n, sigma, nboot=10000):
    x = np.random.rand(nboot, n)
    y = x + np.random.normal(0, sigma, (nboot, n))
    b = np.array([fit_sigmoid(x[i], y[i]) for i in range(nboot)])
    lower, median, upper = np.percentile(b, [2.5, 50, 97.5])
    return pd.Series({'lower': lower, 'median': median, 'upper': upper})


def run(dest, results_path, seed):
    np.random.seed(seed)

    # load human mass responses
    human_mass = pd\
        .read_csv(os.path.join(results_path, "human_mass_responses_by_stimulus.csv"))\
        .groupby('version')\
        .get_group('H')\
        .drop(['version', 'N'], axis=1)\
        .set_index(['stimulus', 'kappa0'])\
        .sortlevel()['median']

    # load model mass responses
    model_mass = pd\
        .read_csv(os.path.join(results_path, "model_mass_responses_by_stimulus.csv"))\
        .set_index(['likelihood', 'counterfactual', 'stimulus', 'kappa0'])\
        .sortlevel()['median']

    results = {}
    for (lh, cf), df in model_mass.groupby(level=['likelihood', 'counterfactual']):
        x = np.asarray(df)
        y = np.asarray(human_mass)
        results[(lh, cf, False)] = bootfit(x, y)
        results[(lh, cf, True)] = bootfit_random(x.size, np.std(x - y))

    results = pd.DataFrame(results).T
    results.index.names = ['likelihood', 'counterfactual', 'random']
    results.to_csv(dest)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path, args.seed)
