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


def plot(dest, results_path, version, block):
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

    fits = pd.read_csv(os.path.join(results_path, "num_samples.csv")).set_index('k')
    best_k = fits['mse'].argmin()

    fig, ax = plt.subplots()
    for k, row in fits.iterrows():
        X = np.array([0, 0.4])
        Y = X * row['slope'] + row['intercept']
        if k == best_k:
            style = '-'
        else:
            style = '--'
        ax.plot(X, Y, style, lw=2, label='$k={:d}$'.format(int(k)))

    ax.plot(model['stddev'], human['stddev'], 'ko')
    ax.set_xlabel('Model stddev')
    ax.set_ylabel('Human stddev')
    ax.set_xlim(0, 0.4)
    ax.legend(loc='best')

    # set figure size
    fig.set_figwidth(6)
    fig.set_figheight(5)
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

    args = parser.parse_args()
    plot(args.to, args.results_path, args.version, args.block)
