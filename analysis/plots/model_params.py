#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(results_path, fig_paths):
    data = pd.read_csv('results/model_params.csv')
    fig, axes = plt.subplots(2, 3, sharey=True, sharex=True)
    version_order = ['H', 'G', 'I']
    model_order = ['static', 'learning']
    colors = {
        'learning': 'r',
        'static': 'b'
    }
    versions = {
        'G': 'Experiment 2a',
        'H': 'Experiment 1',
        'I': 'Experiment 2b'
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
            range=[-0.5, 2.5], normed=True,
            color=colors[model])

        # x = np.linspace(-0.5, 2.5, 100)
        # y = util.kde(x[:, None], B, 0.1)
        # ax.plot(x, np.exp(y), color=colors[m])

        if i == 0:
            ax.set_title(versions[version])
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.9')

    axes[0, 0].legend(loc='best', fontsize=11, frameon=False)

    fig.set_figwidth(10)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
