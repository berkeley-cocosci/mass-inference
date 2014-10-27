#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(results_path, fig_paths):

    model = fig_paths[0]
    fig_paths = fig_paths[1:]

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_trial.csv'))

    llh = pd\
        .read_csv(results_path.joinpath('model_log_lh_ratios.csv'))\
        .set_index(['likelihood', 'version', 'num_trials'])\
        .ix[model]\
        .ix[[('H', 20), ('G', 8), ('I', -1)]]['llhr']\
        .reset_index('num_trials', drop=True)

    colors = {
        'static': 'b',
        'learning': 'r',
    }

    versions = {
        'H': 'Experiment 1',
        'G': 'Experiment 2',
        'I': 'Experiment 3'
    }
    order = ['H', 'G', 'I']

    fig, axes = plt.subplots(1, 3, sharey=True)

    groups = mass_responses.groupby(['version', 'class', 'species'])
    for (version, cls, species), df in groups:
        if species not in (model, 'human'):
            continue

        if species == 'human':
            color = 'k'
            label = 'human'
        elif cls not in colors:
            continue
        else:
            color = colors[cls]
            label = cls

        for kappa0, df2 in df.groupby('kappa0'):
            if kappa0 != 'all':
                continue

            for num, df3 in df2.groupby('num_mass_trials'):
                if version == 'I' and num != -1:
                    continue

                x = np.asarray(df3['trial'], dtype=int)
                y = df3['median']
                yl = df3['lower']
                yu = df3['upper']

                ax = axes[order.index(version)]
                ax.fill_between(x, yl, yu, alpha=0.2, color=color)
                ax.plot(x, y, color=color, lw=2, label=label)

        x = np.sort(df['trial'].unique()).astype(int)
        if version == 'H':
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xticklabels([1, 5, 10, 15, 20])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x)
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel("Trial")
        ax.set_title(versions[version])

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    for lhr, ax in zip(llh, axes):
        l, h = ax.get_xlim()
        ax.text(h, 0.5125,
                r"LLR, learning vs static = {:.2f}".format(lhr),
                horizontalalignment='right', fontsize=9)

    ax = axes[0]
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("Fraction correct")

    axes[-1].legend(
        bbox_to_anchor=[1, 0.1],
        loc='lower right',
        fontsize=9,
        frameon=False)

    fig.set_figwidth(9)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
