#!/usr/bin/env python

"""
Plots distributions of fitted model parameters for each experiment and each
model type (i.e. static and learning).
"""

__depends__ = ["single_model_belief.csv"]

import util
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns


def hist(ax, x, color):
    # plot the histogram
    ax.hist(x, color=color, bins=38, range=[-0.5, 2.5])

    # reformat ytick labels so they show percent, not absolute number
    yticks = np.linspace(0, len(x), 6)
    yticklabels = ["{:.0%}".format(tick / float(len(x))) for tick in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)


def make_legend(ax, colors):
    handles = [
        mpatches.Patch(color=colors['learning'], label='learning'),
        mpatches.Patch(color=colors['static'], label='static')
    ]

    ax.legend(handles=handles, loc='best', fontsize=12, frameon=False)


def plot(dest, results_path):
    # load in the model data, which includes fitted parameters
    data = pd\
        .read_csv(os.path.join(results_path, 'single_model_belief.csv'))\
        .groupby(['fitted', 'counterfactual', 'likelihood'])\
        .get_group((True, True, 'ipe'))\
        .drop_duplicates(['version', 'model', 'pid', 'B'])\
        .set_index(['version', 'model', 'pid'])\
        .sortlevel()

    # double check that there is exactly one parameter for each pid
    assert data.index.is_unique

    # plotting config stuff
    plot_config = util.load_config()["plots"]
    all_colors = plot_config["colors"]
    colors = {
        'static': all_colors[2],
        'learning': all_colors[0]
    }

    # create the figure and plot the histograms
    fig, axes = plt.subplots(2, 3, sharex=True)
    for i, version in enumerate(['H', 'G', 'I']):
        for j, model in enumerate(['static', 'learning']):
            hist(axes[j, i], data.ix[(version, model)]['B'], colors[model])

    # set titles and axis labels
    axes[0, 0].set_title('Experiment 1')
    axes[0, 1].set_title('Experiment 2')
    axes[0, 2].set_title('Experiment 3')
    for ax in axes[:, 0]:
        ax.set_ylabel("% participants")
    for ax in axes[1]:
        ax.set_xlabel(r"Value of $\beta$")

    # make the legend
    make_legend(axes[1, 0], colors)

    # clear top and right axis lines
    sns.despine()

    # set figure size
    fig.set_figwidth(12)
    fig.set_figheight(4)
    plt.draw()
    plt.tight_layout()

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    plot(args.to, args.results_path)
