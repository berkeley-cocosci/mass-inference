#!/usr/bin/env python

"""
Creates a plot of the overall model results for experiment 1. The first subplot
is model vs. human judgments on "will it fall?"; the second two subplots are
model vs. human judgments on "which is heavier?" for both the IPE likelihood and
empirical likelihood.
"""

__depends__ = [
    "human_fall_responses.csv",
    "single_model_fall_responses.csv",
    "fall_response_corrs.csv",
    "human_mass_responses_by_stimulus.csv",
    "model_mass_responses_by_stimulus.csv",
    "mass_responses_by_stimulus_corrs.csv"
]

import util
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns


def errorbar(ax, x, y, ls='', ms=8, marker='o', **kwargs):
    x_lerr = x['median'] - x['lower']
    x_uerr = x['upper'] - x['median']
    y_lerr = y['median'] - y['lower']
    y_uerr = y['upper'] - y['median']
    ax.errorbar(
        x['median'], y['median'], 
        xerr=[x_lerr, x_uerr], 
        yerr=[y_lerr, y_uerr],
        ls=ls, ms=ms, marker=marker,
        **kwargs)


def format_fall_plot(ax, color):
    ax.plot([0, 1], [0, 1], '--', color=color, linewidth=2, zorder=1)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ticklabels = ['0.0', '0.25', '0.50', '0.75', '1.0']
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels)


def format_mass_plot(ax, color):
    xmin = -0.02
    xmax = 1.02
    ymin = -2
    ymax = 102

    ax.plot([xmin, xmax], [50, 50], '--', color=color, linewidth=2, zorder=1)
    ax.plot([0.5, 0.5], [ymin, ymax], '--', color=color, linewidth=2, zorder=1)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ticklabels = ['0.0', '0.25', '0.50', '0.75', '1.0']
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticks([0, 25, 50, 75, 100])


def add_corr(ax, corr):
    pearson = r"$r={median:.2f}$, $95\%\ \mathrm{{CI}}\ [{lower:.2f},\ {upper:.2f}]$"
    corrstr = pearson.format(**corr)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.text(
        xmax - (xmax - xmin) * 0.01, ymin + (ymax - ymin) * 0.035, corrstr,
        horizontalalignment='right', fontsize=10, backgroundcolor='white')


def plot_kappas(ax, model, human, colors, markers):
    errorbar(
        ax, model[-1.0], human[-1.0], 
        color=colors[0], marker=markers[0])
    errorbar(
        ax, model[1.0], human[1.0], 
        color=colors[1], marker=markers[1])


def make_legend(ax, colors, markers):
    handles = []

    for i, kappa in enumerate(['0.1', '10']):
        handles.append(mlines.Line2D(
            [], [],
            color=colors[i], 
            marker=markers[i],
            linestyle='',
            label=r"$\kappa={}$".format(kappa)))

    ax.legend(handles=handles, loc='upper left', fontsize=9, title="True mass ratio")

def plot(dest, results_path, version, counterfactual, query):

    # load human mass responses
    human_mass = pd\
        .read_csv(os.path.join(results_path, "human_mass_responses_by_stimulus.csv"))\
        .groupby('version')\
        .get_group('H')\
        .set_index(['stimulus', 'kappa0'])\
        .sortlevel()\
        .unstack('kappa0')\
        .reorder_levels([1, 0], axis=1) * 100

    # load model mass responses
    model_mass = pd\
        .read_csv(os.path.join(results_path, "model_mass_responses_by_stimulus.csv"))\
        .groupby('counterfactual')\
        .get_group(counterfactual)\
        .set_index(['likelihood', 'stimulus', 'kappa0'])\
        .sortlevel()\
        .unstack('kappa0')\
        .reorder_levels([1, 0], axis=1)

    # load mass correlations
    mass_corrs = pd\
        .read_csv(os.path.join(results_path, "mass_responses_by_stimulus_corrs.csv"))\
        .set_index(['counterfactual', 'version', 'likelihood'])\
        .sortlevel()\
        .ix[(counterfactual, 'H')]

    # color config
    plot_config = util.load_config()["plots"]
    colors = ['.4', '.1']
    markers = ['o', 's']
    lightgrey = plot_config["lightgrey"]
    sns.set_style("white", {'axes.edgecolor': lightgrey})

    # create the figure
    fig, (ax2, ax3) = plt.subplots(1, 2)

    # middle subplot: IPE vs. human (which is heavier?)
    ipe_query = "ipe_" + query
    plot_kappas(ax2, model_mass.ix[ipe_query], human_mass, colors, markers)
    ax2.set_xlabel(r"Ideal observer, $p(\kappa=10|F_t,S_t)$")
    ax2.set_ylabel(r"% participants choosing $\kappa=10$")
    ax2.set_title('(a) Ideal Observer')
    format_mass_plot(ax2, lightgrey)
    add_corr(ax2, mass_corrs.ix[ipe_query])

    # right subplot: empirical vs. human (which is heavier?)
    plot_kappas(ax3, model_mass.ix['empirical'], human_mass, colors, markers)
    ax3.set_xlabel(r"Empirical observer, $p(\kappa=10|F_t,S_t)$")
    #ax3.set_ylabel(r"% participants choosing $\kappa=10$")
    ax3.set_yticklabels([])
    ax3.set_title('(b) Empirical Observer')
    format_mass_plot(ax3, lightgrey)
    add_corr(ax3, mass_corrs.ix['empirical'])

    # create the legend
    make_legend(ax3, colors, markers)

    sns.despine()

    # set figure size
    fig.set_figheight(3.5)
    fig.set_figwidth(7)
    plt.draw()
    plt.tight_layout()

    plt.subplots_adjust(left=0.08, right=0.99)

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
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to plot responses from')
    parser.add_argument(
        '--query',
        default=config['analysis']['query'],
        help='which query for the ipe to use')
    args = parser.parse_args()
    plot(args.to, args.results_path, args.version, args.counterfactual, args.query)
