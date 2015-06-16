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


def plot_sigmoid(ax, model, human, color):
    x = np.linspace(0, 1, 100)
    b = fit_sigmoid(
        np.asarray(model.stack(0)['median']),
        np.asarray(human.stack(0)['median']))
    y = sigmoid(x, b)
    ax.plot(x, y, color=color, lw=2, zorder=2)


def errorbar(ax, x, y, ls='', ms=8, marker='o', **kwargs):
    x_lerr = x['median'] - x['lower']
    x_uerr = x['upper'] - x['median']
    y_lerr = y['median'] - y['lower']
    y_uerr = y['upper'] - y['median']
    ax.errorbar(
        x['median'], y['median'], 
        xerr=[x_lerr, x_uerr], 
        yerr=[y_lerr, y_uerr],
        ls=ls, ms=ms, marker=marker, zorder=3,
        **kwargs)


def format_mass_plot(ax):
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 100])

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
        horizontalalignment='right', fontsize=10)


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

    ax.legend(handles=handles, loc='upper left', fontsize=10, title="True mass ratio")


def plot(dest, results_path, likelihood):

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
        .groupby(['likelihood', 'fitted', 'model', 'version'])\
        .get_group((likelihood, False, 'static', 'H'))\
        .set_index(['counterfactual', 'stimulus', 'kappa0'])\
        .sortlevel()\
        .unstack('kappa0')\
        .reorder_levels([1, 0], axis=1)

    # load mass correlations
    mass_corrs = pd\
        .read_csv(os.path.join(results_path, "mass_responses_by_stimulus_corrs.csv"))\
        .set_index(['likelihood', 'fitted', 'model', 'version', 'counterfactual'])\
        .sortlevel()\
        .ix[(likelihood, False, 'static', 'H')]

    # color config
    plot_config = util.load_config()["plots"]
    palette = plot_config["colors"]
    colors = [palette[0], palette[2]]
    markers = ['o', 's']
    darkgrey = plot_config["darkgrey"]

    # create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2)

    if likelihood == 'empirical':
        likelihood_str = 'Empirical'
    elif likelihood == 'ipe':
        likelihood_str = 'IPE'
    else:
        raise ValueError('unknown likelihood: {}'.format(likelihood))

    # left subplot: regular likelihood
    plot_sigmoid(ax1, model_mass.ix[False], human_mass, color=darkgrey)
    plot_kappas(ax1, model_mass.ix[False], human_mass, colors, markers)
    ax1.set_xlabel(r"{} model, $p(\kappa=10|F_t,S_t)$".format(likelihood_str))
    ax1.set_ylabel(r"% participants choosing $\kappa=10$")
    ax1.set_title("Normal likelihood")
    format_mass_plot(ax1)
    add_corr(ax1, mass_corrs.ix[False])

    # right subplot: counterfactual likelihood
    plot_sigmoid(ax2, model_mass.ix[True], human_mass, color=darkgrey)
    plot_kappas(ax2, model_mass.ix[True], human_mass, colors, markers)
    ax2.set_xlabel(r"{} model, $p(\kappa=10|F_t,S_t)$".format(likelihood_str))
    ax2.set_ylabel(r"% participants choosing $\kappa=10$")
    ax2.set_title("Counterfactual likelihood")
    format_mass_plot(ax2)
    add_corr(ax2, mass_corrs.ix[True])

    # create the legend
    make_legend(ax1, colors, markers)

    # clear the top and right axis lines
    sns.despine()

    # set figure size
    fig.set_figheight(3.5)
    fig.set_figwidth(9)
    plt.draw()
    plt.tight_layout()

    # save
    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    # this is always empirical, regardless of the setting in config.json
    parser.add_argument(
        '--likelihood',
        default='empirical',
        help='which likelihood to use')
    args = parser.parse_args()
    plot(args.to, args.results_path, args.likelihood)
