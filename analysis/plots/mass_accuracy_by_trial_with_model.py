#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util


def plot(results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_trial.csv'))

    colors = {
        'human': 'k',
        'empirical': 'r',
        'ipe': 'b',
    }

    versions = ['H', 'G', 'I']

    fig, axes = plt.subplots(1, 3, sharey=True)

    groups = mass_responses.groupby(['version', 'class', 'species'])
    for (version, cls, species), df in groups:
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

                ax = axes[versions.index(version)]
                ax.fill_between(x, yl, yu, alpha=0.3, color=colors[species])
                ax.plot(x, y, color=colors[species], lw=2, label=num,
                        marker='o', markersize=4)

        x = np.sort(df['trial'].unique()).astype(int)
        if version == 'H':
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_xticklabels([1, 5, 10, 15, 20])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x)
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel("Trial")
        ax.set_title("Experiment %d" % (versions.index(version) + 1))

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

    ax = axes[0]
    ax.set_ylim(0.5, 1)
    ax.set_ylabel("Fraction correct")

    fig.set_figwidth(9)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_by_trial_with_model.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
