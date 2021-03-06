#!/usr/bin/env python

"""
Creates a plot of model vs. human mass responses using both the
non-counterfactual likelihoods, and the counterfactual likelihoods.
"""

__depends__ = [
    "human_mass_responses_by_stimulus.csv",
    "model_mass_responses_by_stimulus.csv",
    "mass_responses_by_stimulus_corrs.csv"
]

import util
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.optimize as optim


def sigmoid(x, b, offset=0.5, scale=100.0):
    return scale / (1.0 + np.exp(-(b * (x - offset))))


def fit_sigmoid(xarr, yarr):
    def fit(b):
        pred = sigmoid(xarr, b)
        diff = np.mean(np.abs(yarr - pred))
        return diff

    res = optim.minimize_scalar(fit)
    return res['x']


def errorbar(ax, x, y, ls='', ms=6, marker='o', **kwargs):
    x_lerr = x['median'] - x['lower']
    x_uerr = x['upper'] - x['median']
    y_lerr = y['median'] - y['lower']
    y_uerr = y['upper'] - y['median']
    util.notfilled_errorbar(
        ax, x['median'], y['median'],
        xerr=[x_lerr, x_uerr],
        yerr=[y_lerr, y_uerr],
        ls=ls, ms=ms, marker=marker, zorder=3, linewidth=1,
        **kwargs)

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
        xmax - (xmax - xmin) * 0.01, ymin + (ymax - ymin) * 0.025, corrstr,
        horizontalalignment='right')


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
            markeredgecolor=colors[i], 
            markerfacecolor='white',
            markeredgewidth=1,
            marker=markers[i],
            linestyle='',
            label=r"$\kappa={}$".format(kappa)))

    ax.legend(handles=handles, loc='upper left', title="True mass ratio")


def plot(dest, results_path, query):

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
        .set_index(['likelihood', 'counterfactual', 'stimulus', 'kappa0'])\
        .sortlevel()\
        .unstack('kappa0')\
        .reorder_levels([1, 0], axis=1)

    # load mass correlations
    mass_corrs = pd\
        .read_csv(os.path.join(results_path, "mass_responses_by_stimulus_corrs.csv"))\
        .set_index(['version', 'likelihood', 'counterfactual'])\
        .sortlevel()\
        .ix['H']

    # color config
    plot_config = util.load_config()["plots"]
    #palette = plot_config["colors"]
    #colors = [palette[0], palette[2]]
    colors = ['.4', '.1']
    markers = ['o', 's']
    lightgrey = plot_config["lightgrey"]
    sns.set_style("white", {'axes.edgecolor': lightgrey})

    # create the figure
    fig, ax = plt.subplots()

    plot_kappas(ax, model_mass.ix[('empirical', True)], human_mass, colors, markers)
    ax.set_xlabel(r"Empirical observer, $p(\kappa=10|F_t,S_t)$")
    ax.set_ylabel(r"% participants choosing $\kappa=10$")
    format_mass_plot(ax, lightgrey)
    add_corr(ax, mass_corrs.ix[('empirical', True)])

    # create the legend
    make_legend(ax, colors, markers)

    # clear the top and right axis lines
    sns.despine()

    # set figure size
    fig.set_figheight(3)
    fig.set_figwidth(3.5)
    plt.draw()
    plt.tight_layout()

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--query',
        default=config['analysis']['query'],
        help='which query for the ipe to use')
    args = parser.parse_args()
    plot(args.to, args.results_path, args.query)
