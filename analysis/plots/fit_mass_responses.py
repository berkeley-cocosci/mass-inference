#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util


def plot(results_path, fig_paths):

    responses = pd\
        .read_csv(results_path.joinpath('mass_responses_by_stimulus.csv'))\
        .groupby('version')\
        .get_group('H')
    responses = responses.ix[responses['stimulus'] != 'prior']

    params = pd\
        .read_csv(results_path.joinpath("fit_mass_responses.csv"))\
        .set_index('model')['median']

    colors = {
        -1.0: 'r',
        1.0: 'b'
    }

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)

    xspace = np.linspace(0, 1, 100)

    for kappa0, df in responses.groupby('kappa0'):
        model = df.groupby('species').get_group('ipe')
        human = df.groupby('species').get_group('human')

        x = model['median']
        xl = x - model['lower']
        xu = model['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax1.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)
        ax1.plot(xspace, util.sigmoid(xspace, params['ipe']),
                 'k--', linewidth=2)

        ax2.errorbar(util.sigmoid(x, params['ipe']),
                     y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

        model = df.groupby('species').get_group('empirical')
        human = df.groupby('species').get_group('human')

        x = model['median']
        xl = x - model['lower']
        xu = model['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax3.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)
        ax3.plot(xspace, util.sigmoid(xspace, params['empirical']),
                 'k--', linewidth=2)

        ax4.errorbar(util.sigmoid(x, params['empirical']),
                     y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

    ax1.set_ylabel("Human")
    ax1.set_xlabel("IPE")
    ax2.set_xlabel("Transformed IPE")
    ax3.set_xlabel("Empirical IPE")
    ax4.set_xlabel("Transformed Empirical IPE")

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.9')

    fig.set_figwidth(15)
    fig.set_figheight(3.5)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
