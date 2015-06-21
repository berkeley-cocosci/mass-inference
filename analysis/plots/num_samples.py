#!/usr/bin/env python

"""
Plots the standard deviation of the model vs. human for "will it fall?" as well
as best fitting lines for several different k values (number of samples).
"""

__depends__ = [
    "fall_response_corrs.csv",
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

    fall_corrs = pd\
        .read_csv(os.path.join(results_path, "fall_response_corrs.csv"))\
        .set_index(['block', 'X', 'Y'])\
        .sortlevel()\
        .ix[(block, query, 'Human')]

    colors = sns.color_palette("cubehelix", 7)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # means
    mlerr = model['median'] - model['lower']
    muerr = model['upper'] - model['lower']
    hlerr = human['median'] - human['lower']
    huerr = human['upper'] - human['median']
    ax1.plot([-0.03, 1.03], [-0.03, 1.03], '--', color=config['plots']['darkgrey'])
    ax1.errorbar(
        model['median'], human['median'],
        xerr=[mlerr, muerr], yerr=[hlerr, huerr],
        color='k', linestyle='', marker='o')
    ax1.set_xlabel(r"Ideal observer model, $p(F_t|S_t)$")
    ax1.set_ylabel(r"Normalized human judgments")
    ax1.set_title('(a) Mean judgments')
    ax1.set_xlim(-0.03, 1.03)
    ax1.set_ylim(-0.03, 1.03)

    pearson = r"$r={median:.2f}$, $95\%\ \mathrm{{CI}}\ [{lower:.2f},\ {upper:.2f}]$"
    corrstr = pearson.format(**fall_corrs)
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    ax1.text(
        xmax - (xmax - xmin) * 0.01, ymin + (ymax - ymin) * 0.035, corrstr,
        horizontalalignment='right', fontsize=10, backgroundcolor='white')

    # variances
    for k, row in fits.iterrows():
        X = np.array([-0.005, 0.205])
        Y = X * row['slope'] + row['intercept']
        if k == best_k:
            style = '-'
        else:
            style = '--'
        ax2.plot(X, Y, style, lw=2, label='$k={:d}$'.format(int(k)), color=colors[k - 1])

    ax2.plot(model['stddev']**2, human['stddev']**2, 'ko')
    ax2.set_xlabel(r'Model variance ($\sigma_\mathrm{sims}^2$)')
    ax2.set_ylabel(r'Human variance ($\sigma_\mathrm{judgments}^2$)')
    ax2.set_xlim(-0.005, 0.205)
    ax2.set_ylim(-0.005, 0.205)
    ax2.set_title("(b) Variance of judgments")
    ax2.legend(loc='best', ncol=2)

    sns.despine()

    # set figure size
    fig.set_figwidth(7)
    fig.set_figheight(3.5)
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
