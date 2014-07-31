#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_paths):

    responses = pd\
        .read_csv(results_path.joinpath('mass_responses_by_stimulus.csv'))\
        .groupby('version')\
        .get_group('H')

    colors = {
        -1.0: 'r',
        1.0: 'b'
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

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
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

        x = empirical['median']
        xl = x - empirical['lower']
        xu = empirical['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax2.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

        x = empirical['median']
        xl = x - empirical['lower']
        xu = empirical['upper'] - x

        y = ipe['median']
        yl = y - ipe['lower']
        yu = ipe['upper'] - y

        ax3.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

    ax1.set_title("IPE vs. Human")
    ax2.set_title("Empirical IPE vs. Human")
    ax3.set_title("Empirical IPE vs. IPE")

    ax1.set_xlabel("IPE")
    ax1.set_ylabel("Human")
    ax2.set_xlabel("Empirical IPE")
    ax2.set_ylabel("Human")
    ax3.set_xlabel("Empirical IPE")
    ax3.set_ylabel("IPE")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.9')

    fig.set_figwidth(12)
    fig.set_figheight(3.5)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
