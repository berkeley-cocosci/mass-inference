#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


def plot(results_path, fig_paths):
    data = pd.read_csv('results/model_params.csv')
    fig, axes = plt.subplots(2, 3, sharex=True)
    version_order = ['H', 'G', 'I']
    model_order = ['static', 'learning']
    colors = {
        'learning': util.colors[2],
        'static': util.colors[0]
    }
    versions = {
        'G': 'Experiment 2',
        'H': 'Experiment 1',
        'I': 'Experiment 3'
    }

    for (version, model), params in data.groupby(['version', 'model']):
        if model not in model_order:
            continue

        j = version_order.index(version)
        i = model_order.index(model)
        ax = axes[i, j]

        B = np.asarray(params['B'].dropna())
        ax.hist(
            B, label=model, bins=38,
            range=[-0.5, 2.5], normed=False,
            color=colors[model])

        if i == 0:
            ax.set_title(versions[version])
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)

        ax.set_yticks(np.linspace(0, len(B), 6))
        yticks = ax.get_yticks()
        if j == 0:
            yticklabels = ["{:.1f}".format(tick / float(len(B))) for tick in yticks]
            ax.set_yticklabels(yticklabels)
            ax.set_ylabel("% participants")
        else:
            ax.set_yticklabels([])

        if i == 1:
            ax.set_xlabel(r"Value of $\beta$")

    learning = mpatches.Patch(color=colors['learning'], label='learning')
    static = mpatches.Patch(color=colors['static'], label='static')
    axes[1, 0].legend(
        handles=[static, learning], loc='best', fontsize=12, frameon=False)

    fig.set_figwidth(12)
    fig.set_figheight(4)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
