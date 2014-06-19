#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_trial.csv'))\
        .groupby(['class', 'species'])\
        .get_group(('chance', 'human'))

    colors = {
        -1.0: 'r',
        1.0: 'b'
    }

    fig, axes = plt.subplots(1, 3)

    for i, (version, df) in enumerate(mass_responses.groupby('version')):
        for kappa0, df2 in df.groupby('kappa0'):
            x = df2['trial']
            y = df2['median']
            yl = df2['lower']
            yu = df2['upper']

            ax = axes[i]
            ax.fill_between(x, yl, yu, alpha=0.3, color=colors[kappa0])
            ax.plot(x, y, color=colors[kappa0], lw=2)
            ax.set_xlim(1, 20)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Trial")
            ax.set_ylabel("Fraction correct")
            ax.set_title(version)

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
