#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(results_path, fig_paths):

    responses = pd\
        .read_csv(results_path.joinpath('mass_responses_by_stimulus.csv'))\
        .groupby('version')\
        .get_group('H')

    colors = {
        -1.0: util.colors[0],
        1.0: util.colors[2]
    }

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for kappa0, df in responses.groupby('kappa0'):
        empirical = df\
            .groupby('species')\
            .get_group('empirical')
        ipe = df\
            .groupby('species')\
            .get_group('ipe')
        human = df\
            .groupby('species')\
            .get_group('human')

        x = ipe['median']
        xl = x - ipe['lower']
        xu = ipe['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax1.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor=util.darkgrey,
                     label=r"$\kappa_0=%s$" % (10 ** kappa0))

        x = empirical['median']
        xl = x - empirical['lower']
        xu = empirical['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax2.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor=util.darkgrey)

    ax1.set_title("IPE vs. Human")
    ax2.set_title("Empirical vs. Human")

    ax1.set_xlabel("IPE")
    ax1.set_ylabel("Human")
    ax2.set_xlabel("Empirical")
    ax2.set_ylabel("Human")

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    ax1.legend(loc='upper left', fontsize=11, frameon=False)

    fig.set_figwidth(8)
    fig.set_figheight(3.5)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
