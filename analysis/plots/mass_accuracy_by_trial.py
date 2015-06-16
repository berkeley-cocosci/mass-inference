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


def timeseries(ax, x, color, label=None):
    ax.fill_between(
        x['trial'], x['lower'], x['upper'],
        color=color, alpha=0.3)
    ax.plot(
        x['trial'], x['median'], 
        lw=2, marker='o', ms=6, color=color, label=label)


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
    colors = plot_config["colorcircle"][::-1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    # left subplot: experiment 2
    timeseries(ax1, responses.ix['G'], darkgrey)
    ax1.set_title('Experiment 1b\n')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Fraction correct')
    ax1.set_xticks([1, 2, 3, 4, 6, 9, 14, 20])
    ax1.set_xlim([1, 20])
    ax1.set_ylim(0.5, 1)

    # middle subplot: experiment 3 (between subjects)
    timeseries(ax2, responses.ix[('I', -1)], darkgrey)
    ax2.set_title('Experiment 2\n(between subjects)')
    ax2.set_xlabel('Trial')
    ax2.set_xticks([1, 2, 3, 5, 10])
    ax2.set_xlim([1, 10])

    # right subplot: experiment 3 (within subjects)
    for i in range(1, 6):
        timeseries(
            ax3, responses.ix[('I', i)], colors[i - 1], 
            label="{} trials".format(i))
    ax3.set_title('Experiment 2\n(within subjects)')
    ax3.set_xlabel('Trial')
    ax3.set_xticks([1, 2, 3, 5, 10])
    ax3.set_xlim([1, 10])

    # draw the legend
    ax3.legend(loc='lower center', fontsize=10, ncol=2, frameon=False)

    # clear top and right axis lines
    sns.despine()

    # set figure size
    fig.set_figwidth(9)
    fig.set_figheight(3.25)
    plt.draw()
    plt.tight_layout()

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    plot(args.to, args.results_path)
