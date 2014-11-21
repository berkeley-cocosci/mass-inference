#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(results_path, fig_paths):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_stimulus.csv'))\
        .groupby(['version', 'species'])\
        .get_group(('H', 'human'))

    colors = {
        -1.0: util.colors[0],
        1.0: util.colors[2]
    }

    fig, ax = plt.subplots()

    for i, (kappa0, df) in enumerate(mass_responses.groupby('kappa0')):
        x = i * 0.3 + np.arange(len(df['stimulus']))
        y = df['median']
        yl = y - df['lower']
        yu = df['upper'] - y

        ax.bar(x, y, yerr=[yl, yu],
               color=colors[kappa0],
               ecolor=util.darkgrey,
               width=0.3,
               label='kappa=%s' % kappa0)

    ax.legend(loc='best')
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Stimulus")
    ax.set_ylabel("Fraction correct")

    util.clear_right(ax)
    util.clear_top(ax)
    util.outward_ticks(ax)

    fig.set_figwidth(8)
    fig.set_figheight(4)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
