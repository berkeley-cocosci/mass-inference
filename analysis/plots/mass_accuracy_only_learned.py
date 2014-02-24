#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_responses_only_learned.csv'))\
        .groupby('species')\
        .get_group('human')

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for i, (kappa0, df) in enumerate(mass_responses.groupby('kappa0')):
        x = df['trial']
        y = df['median']
        yl = df['lower']
        yu = df['upper']

        ax = axes[i]
        ax.fill_between(x, yl, yu, alpha=0.3, color='k')
        ax.plot(x, y, color='k', lw=2)
        ax.set_title(r"$\kappa_0=%.1f$" % kappa0)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Fraction correct")

    fig.set_figwidth(8)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_only_learned.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
