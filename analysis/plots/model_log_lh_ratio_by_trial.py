#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(results_path, fig_paths):
    llr = pd.read_csv(results_path.joinpath(
        "model_log_lh_ratio_by_trial.csv"))
    llr = llr.pivot('trial', 'version', 'llhr').cumsum()

    linestyles = {
        'G': '--',
        'I (across)': '-',
        'I (within)': ':'
    }

    labels = {
        'G': 'Experiment 2',
        'I (across)': 'Experiment 3 (across subjects)',
        'I (within)': 'Experiment 3 (within subjects)'
    }

    order = ['G', 'I (within)', 'I (across)']

    fig, ax = plt.subplots()

    for col in order:
        df = llr[col].dropna()
        ls = linestyles[col]
        if ls == ':':
            lw = 5
        else:
            lw = 3

        ax.plot(
            df.index, df, label=labels[col],
            lw=lw, color='k', ls=ls)

    ax.hlines([0], 1, 20, color='k', linestyle='dotted')
    ax.legend(loc='lower left', fontsize=10, frameon=False)
    ax.set_xlim(1, 20)
    ax.set_ylim(-125, 75)
    ax.set_ylabel("Cumulative evidence ($D$)")
    ax.set_xlabel("Trial")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xticklabels([1, 5, 10, 15, 20])

    util.outward_ticks(ax)
    util.clear_right(ax)
    util.clear_top(ax)

    fig.set_figwidth(4)
    fig.set_figheight(3.5)

    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
