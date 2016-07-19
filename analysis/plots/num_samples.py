#!/usr/bin/env python

"""
Plots the standard deviation of the model vs. human for "will it fall?" as well
as best fitting lines for several different k values (number of samples).
"""

__depends__ = [
    "human_fall_responses.csv",
    "single_model_fall_responses.csv",
    "num_samples.csv"
]

import util
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colormaps import _viridis_data as viridis


def plot(dest, results_path, version, block, query):
    human = pd.read_csv(os.path.join(results_path, "human_fall_responses.csv"))
    human = human\
        .groupby(['version', 'block'])\
        .get_group((version, block))\
        .set_index(['kappa0', 'stimulus'])\
        .sortlevel()

    model = pd.read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))
    model = model\
        .groupby(['query', 'block'])\
        .get_group((query, block))\
        .set_index(['kappa0', 'stimulus'])\
        .sortlevel()
    model = model.ix[human.index]

    fits = pd.read_csv(os.path.join(results_path, "num_samples.csv"))\
        .groupby(['version', 'query', 'block'])\
        .get_group((version, query, block))\
        .set_index('k')
    best_k = fits['mse'].argmin()

    colors = np.array(viridis)[np.linspace(0, len(viridis) - 1, 7).astype(int)]

    fig, ax = plt.subplots()

    # variances
    lines = []
    labels = []
    for k, row in fits.iterrows():
        X = np.array([-0.005, 0.205])
        Y = X * row['slope'] + row['intercept']
        if k == best_k:
            style = '-'
        else:
            style = '--'
        l, = ax.plot(X, Y, style, lw=1, color=colors[k - 1])
        lines.append(l)
        labels.append('$n={:d}$'.format(int(k)))

    ax.plot(
        model['stddev']**2, human['stddev']**2, 'ko',
        markeredgecolor=config['plots']['darkgrey'],
        markerfacecolor='white',
        markeredgewidth=1)
    ax.set_xlabel(r'Model variance ($\sigma_\mathrm{sims}^2$)')
    ax.set_ylabel(r'Human variance ($\sigma_\mathrm{judgments}^2$)')
    ax.set_xlim(-0.005, 0.205)
    ax.set_ylim(-0.005, 0.205)
    ax.legend(lines, labels, loc='best', ncol=2)

    sns.despine()

    # set figure size
    fig.set_figwidth(3.25)
    fig.set_figheight(3.25)
    plt.draw()
    plt.tight_layout()

    # save
    for pth in dest:
        util.save(pth, close=False)

if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to plot responses from')
    parser.add_argument(
        '--block',
        default='B',
        help='which block of the experiment to plot responses from')
    parser.add_argument(
        '--query',
        default=config['analysis']['query'],
        help='which ipe query to use')

    args = parser.parse_args()
    plot(args.to, args.results_path, args.version, args.block, args.query)
