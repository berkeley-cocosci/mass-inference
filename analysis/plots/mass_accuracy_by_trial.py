#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import util


def plot(results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_trial.csv'))\
        .groupby(['class', 'species'])\
        .get_group(('chance', 'human'))

    colors = {
        8: 'k',
        20: 'k',
        -1: 'k',
        5: 'r',
        4: 'y',
        3: 'g',
        2: 'b',
        1: 'm'
    }

    versions = ['H', 'G', 'I']

    fig, axes = plt.subplots(1, 3)

    for version, df in mass_responses.groupby('version'):
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
                ax.fill_between(x, yl, yu, alpha=0.3, color=colors[num])
                ax.plot(x, y, color=colors[num], lw=2, label=num)
                if version == 'H':
                    ax.set_xticks([1, 5, 10, 15, 20])
                    ax.set_xticklabels([1, 5, 10, 15, 20])
                else:
                    ax.set_xticks(x)
                    ax.set_xticklabels(x)
                ax.set_xlim(x.min(), x.max())
                ax.set_ylim(0.5, 1)
                ax.set_xlabel("Trial")
                ax.set_ylabel("Fraction correct")
                ax.set_title("Experiment %d" % (versions.index(version) + 1))

    fig.set_figwidth(15)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_by_trial.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
