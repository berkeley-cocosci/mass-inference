#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def plot(results_path, fig_paths):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_trial.csv'))

    llh = pd\
        .read_csv(results_path.joinpath('model_log_lh_ratios.csv'))\
        .set_index(['version', 'num_trials'])\
        .ix[[('H', 20), ('G', 8), ('I', -1)]][['llhr', 'likelihood']]\
        .reset_index('num_trials', drop=True)\
        .set_index('likelihood', append=True)['llhr']\
        .unstack('likelihood')

    palette = sns.color_palette("Dark2")
    colors = {
        'static': palette[0],
        'learning': palette[1],
    }

    linestyles = {
        'empirical': '-',
        'ipe': '--'
    }

    versions = {
        'H': 'Experiment 1',
        'G': 'Experiment 2',
        'I': 'Experiment 3'
    }
    order = ['H', 'G', 'I']

    fig, axes = plt.subplots(1, 3, sharey=True)
    lines = {}

    groups = mass_responses.groupby(['version', 'class', 'species'])
    for (version, cls, species), df in groups:
        if species == 'human':
            color = util.darkgrey
            label = 'Human'
            ls = '-'
        elif cls not in colors:
            continue
        else:
            color = colors[cls]
            label = "{} model,\n".format(cls.capitalize())
            if species == 'ipe':
                label = "{}IPE likelihod".format(label)
            elif species == "empirical":
                label = "{}Emp. likelihood".format(label)
            ls = linestyles[species]

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
                line, = ax.plot(x, y, color=color, lw=2, ls=ls)
                lines[label] = line

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

    for version in llh.T:
        ax = axes[order.index(version)]
        l, h = ax.get_xlim()
        lhr = llh.T[version]
        label = "{} LLR = {:.2f}"
        text = "{}\n{}".format(
            label.format("Empirical", lhr["empirical"]),
            label.format("IPE", lhr["ipe"]))
        ax.text(h, 0.5125, text, horizontalalignment='right', fontsize=9)

    ax = axes[0]
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("Fraction correct")

    labels, lines = zip(*sorted(lines.items()))

    axes[-1].legend(
        lines, labels,
        bbox_to_anchor=[1.05, 0.5],
        loc='center left',
        fontsize=9,
        frameon=False)

    fig.set_figwidth(9)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(right=0.825)

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
