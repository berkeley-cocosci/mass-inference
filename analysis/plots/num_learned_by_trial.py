#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util


def plot(results_path, fig_paths):

    nlearned = pd.read_csv(
        results_path.joinpath('num_learned_by_trial.csv'))

    colors = {
        8: 'k',
        20: 'k',
        5: 'r',
        4: 'y',
        3: 'g',
        2: 'b',
        1: 'm'
    }

    linestyles = {
        'all': '-',
        'chance': '--'
    }

    markers = {
        'all': 'o',
        'chance': ''
    }

    versions = ['H', 'G', 'I']

    fig, axes = plt.subplots(1, 3, sharey=True)
    for version, df in nlearned.groupby('version'):
        for kappa0, df2 in df.groupby('kappa0'):
            if kappa0 not in linestyles:
                continue

            for num, df3 in df2.groupby('num_mass_trials'):
                if version == 'I' and kappa0 == 'chance' and num != 5:
                    continue
                if kappa0 == 'chance':
                    color = 'k'
                    label = 'chance'
                else:
                    color = colors[num]
                    label = '%d trials' % num

                x = np.asarray(df3['trial'], dtype=int)
                y = df3['median']
                yl = df3['lower']
                yu = df3['upper']

                ax = axes[versions.index(version)]
                ax.fill_between(
                    x, yl, yu,
                    alpha=0.3,
                    color=color)
                ax.plot(
                    x, y,
                    color=color,
                    lw=2,
                    label=label,
                    linestyle=linestyles[kappa0],
                    marker=markers[kappa0],
                    markersize=4)

        x = np.sort(df['trial'].unique()).astype(int)
        ax.set_xlim(x.min(), x.max())
        if version == 'H':
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xticklabels([1, 5, 10, 15, 20])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x)
        ax.set_xlabel("Trial")
        ax.set_title("Experiment %d" % (versions.index(version) + 1))

        if version == 'I':
            ax.legend(loc='lower center', fontsize=10, ncol=2, frameon=False)

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    ax = axes[0]
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of participants")

    fig.set_figwidth(9)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
