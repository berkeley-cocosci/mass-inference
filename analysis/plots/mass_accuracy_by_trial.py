#!/usr/bin/env python

"""
Plots human accuracy on "which is heavier?" as a function of trial for
experiments 2 and 3 (both between subjects and within subjects for experiment
3).
"""

__depends__ = ["human_mass_accuracy_by_trial.csv"]

import util
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from colormaps import _viridis_data as viridis


def timeseries(ax, x, color, label=None, marker='o', markersize=6, fill=True):
    if fill:
        ax.fill_between(
            x['trial'], x['lower'], x['upper'],
            color=color, alpha=0.3)
    ax.plot(
        x['trial'], x['median'], 
        lw=1, marker=marker, ms=markersize, color=color,
        markeredgecolor=color, markerfacecolor='none',
        markeredgewidth=1, label=label)


def plot(dest, results_path):

    # load in the responses
    responses = pd\
        .read_csv(os.path.join(results_path, 'human_mass_accuracy_by_trial.csv'))\
        .groupby('kappa0')\
        .get_group('all')\
        .set_index(['version', 'num_mass_trials'])

    # load in colors
    plot_config = util.load_config()["plots"]
    darkgrey = plot_config["darkgrey"]
    #colors = np.ones((5, 3)) * np.linspace(0.1, 0.6, 5)[:, None]
    colors = np.array(viridis)[np.linspace(0, len(viridis) - 1, 6).astype(int)[:-1]]
    markers = ['s', 'd', '*', '^', 'o']
    markersizes = [6, 7, 10, 7, 6]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    # left subplot: experiment 2
    timeseries(ax1, responses.ix['G'], darkgrey)
    ax1.set_title('(a) Experiment 3.2')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Proportion correct')
    ax1.set_xticks([1, 2, 3, 4, 6, 9, 14, 20])
    ax1.set_xlim([1, 20.5])
    ax1.set_ylim(0.5, 1)

    # middle subplot: experiment 3 (between subjects)
    timeseries(ax2, responses.ix[('I', -1)], darkgrey)
    ax2.set_title('(b) Exp. 3.3, between subjs.')
    ax2.set_xlabel('Trial')
    ax2.set_xticks([1, 2, 3, 5, 10])
    ax2.set_xlim([1, 10.25])

    # right subplot: experiment 3 (within subjects)
    for i in range(5, 0, -1):
        timeseries(
            ax3, responses.ix[('I', i)], colors[i - 1],
            label="{} trials".format(i),
            marker=markers[i - 1],
            markersize=markersizes[i - 1],
            fill=False)
    ax3.set_title('(c) Exp. 3.3, within subjs.')
    ax3.set_xlabel('Trial')
    ax3.set_xticks([1, 2, 3, 5, 10])
    ax3.set_xlim([1, 10.25])

    # draw the legend
    ax3.legend(loc='lower center', ncol=2, frameon=False)

    # clear top and right axis lines
    sns.despine()

    # set figure size
    fig.set_figwidth(6.5)
    fig.set_figheight(2.35)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, right=0.99, left=0.08)

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    plot(args.to, args.results_path)
