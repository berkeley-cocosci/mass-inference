#!/usr/bin/env python

"""
Plots distributions of fitted model parameters for each experiment and each
model type (i.e. static and learning).
"""

__depends__ = ["model_belief_by_trial_fit.csv"]

import util
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn as sns


def hist(ax, x, color):
    ymax = len(x) * 1.02
    #ax.plot([0, 0], [0, ymax], '-', color='0.6', zorder=1)
    #ax.plot([1, 1], [0, ymax], ':', color='0.6', zorder=1)
    ax.fill_between([0, 1], [0, 0], [ymax, ymax], color='0.8', zorder=1)

    med = x.median()
    ax.plot([med, med], [0, ymax], ':', color=color, zorder=2, lw=1)

    # plot the histogram
    bins = np.arange(-0.7, 2.8, 0.2)
    ax.hist(x, color=color, bins=bins, zorder=10)

    # reformat ytick labels so they show percent, not absolute number
    yticks = np.linspace(0, len(x), 6)
    yticklabels = ["{:.0%}".format(tick / float(len(x))) for tick in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(0, ymax)
    ax.set_xlim(bins.min(), bins.max())

def make_legend(ax, colors):
    handles = [
        mpatches.Patch(color=colors['learning'], label='learning'),
        mpatches.Patch(color=colors['static'], label='static')
    ]

    ax.legend(handles=handles, loc='best', frameon=False)


def plot(dest, results_path, counterfactual, likelihood):
    if likelihood == 'ipe':
        likelihood = 'ipe_' + util.get_query()

    # load in the model data, which includes fitted parameters
    data = pd\
        .read_csv(os.path.join(results_path, 'model_belief_by_trial_fit.csv'))\
        .groupby(['fitted', 'counterfactual', 'likelihood'])\
        .get_group((True, counterfactual, likelihood))\
        .drop_duplicates(['version', 'model', 'pid', 'B'])\
        .set_index(['version', 'model', 'pid'])\
        .sortlevel()

    # double check that there is exactly one parameter for each pid
    assert data.index.is_unique

    # plotting config stuff
    plot_config = util.load_config()["plots"]
    all_colors = plot_config["colors"]
    colors = {
        'static': all_colors[1],
        'learning': all_colors[0]
    }

    # create the figure and plot the histograms
    fig, axes = plt.subplots(2, 3, sharex=True)
    for i, version in enumerate(['H', 'G', 'I']):
        for j, model in enumerate(['static', 'learning']):
            hist(axes[j, i], data.ix[(version, model)]['B'], plot_config['darkgrey'])

    # set titles and axis labels
    for i in range(3):
        axes[0, i].set_title('Experiment 3.{}'.format(i+1), y=1.05)
    for ax in axes[:, 0]:
        ax.set_ylabel("% participants")
    for ax in axes[1]:
        ax.set_xlabel(r"Best fit learning rate ($\beta$)")

    for i, label in enumerate(['Static', 'Learning']):
        mid = sum(axes[i, 0].get_ylim()) / 2.0
        axes[i, 0].text(
            -2.6, mid, label,
            rotation=90,
            fontsize=12, # same as title font size
            verticalalignment='center')

    # clear top and right axis lines
    sns.despine()

    # set figure size
    fig.set_figwidth(6.5)
    fig.set_figheight(3.25)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(left=0.14)

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    if config['analysis']['counterfactual']:
        parser.add_argument(
            '--no-counterfactual',
            action='store_false',
            dest='counterfactual',
            default=True,
            help="don't plot the counterfactual likelihoods")
    else:
        parser.add_argument(
            '--counterfactual',
            action='store_true',
            dest='counterfactual',
            default=False,
            help='plot the counterfactual likelihoods')
    parser.add_argument(
        '--likelihood',
        default=config['analysis']['likelihood'],
        help='which version of the likelihood to plot')
    args = parser.parse_args()
    plot(args.to, args.results_path, args.counterfactual, args.likelihood)
