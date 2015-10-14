#!/usr/bin/env python

"""
Plots model vs human responses to "will it fall?" for a particular experiment
version and block. Each figure has three subplots:

    1. human (kappa=-1.0) vs human (kappa=1.0)
    2. model (mass-sensitive) vs human
    3. model (mass-insensitive) vs human

"""

__depends__ = ["human_fall_responses.csv", "single_model_fall_responses.csv"]

import util
import os
import matplotlib.pyplot as plt
import pandas as pd


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

def plot_connected(ax, x0, x1, y0, y1, colors, labels, plot_config):
    ax.plot(
        [x0['median'], x1['median']], 
        [y0['median'], y1['median']], 
        ls='-', color=plot_config["darkgrey"])

    errorbar(ax, x0, y0, color=colors[0], label=labels[0])
    errorbar(ax, x1, y1, color=colors[1], label=labels[1])


def plot(dest, results_path, version, block):

    # various config stuff
    config = util.load_config()
    plot_config = config["plots"]
    colors = [plot_config["colors"][0], plot_config["colors"][2]]
    labels = [r"$\kappa_0=%.1f$" % 10 ** kappa0 for kappa0 in [-1.0, 1.0]]

    # read in the human data
    human = pd\
        .read_csv(os.path.join(results_path, "human_fall_responses.csv"))\
        .groupby(['version', 'block'])\
        .get_group((version, block))\
        .set_index(['stimulus', 'kappa0'])\
        .unstack('kappa0')\
        .reorder_levels([1, 0], axis=1)

    # read in the model data
    model = pd\
        .read_csv(os.path.join(results_path, "single_model_fall_responses.csv"))\
        .groupby(['query', 'block'])\
        .get_group((util.get_query(), block))\
        .set_index(['stimulus', 'kappa0'])\
        .unstack('kappa0')\
        .reorder_levels([1, 0], axis=1)

    # create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    # left subplot: human vs. human, for both values of the mass ratio
    errorbar(ax1, human[-1.0], human[1.0], color=plot_config["darkgrey"])
    ax1.set_xlabel(r"Human ($\kappa_0=0.1$)")
    ax1.set_ylabel(r"Human ($\kappa_0=10.0$)")

    # middle subplot: mass sensitive model
    ax2.plot([0, 1], [0, 1], '--', color=plot_config["darkgrey"], alpha=0.5)
    plot_connected(
        ax2, model[-1.0], model[1.0], human[-1.0], human[1.0],
        colors, labels, plot_config)
    ax2.set_xlabel("Mass-sensitive IPE")
    ax2.set_ylabel("Human")

    # right subplot: mass-insensitive model
    ax3.plot([0, 1], [0, 1], '--', color=plot_config["darkgrey"], alpha=0.5)
    plot_connected(
        ax3, model[0.0], model[0.0], human[-1.0], human[1.0],
        colors, labels, plot_config)
    ax3.set_xlabel("Mass-insensitive IPE")
    ax3.set_ylabel("Human")

    # make the legend
    ax3.legend(loc='lower right', frameon=False)

    # set axis limits
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # adjust figure size
    fig.set_figwidth(6.5)
    fig.set_figheight(1.9)
    plt.draw()
    plt.tight_layout()

    # save to file
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
