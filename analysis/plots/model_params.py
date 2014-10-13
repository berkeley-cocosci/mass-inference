#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(results_path, fig_paths):
    data = pd.read_csv('results/model_params.csv')
    fig, axes = plt.subplots(1, 3, sharey=True)
    order = ['G', 'H', 'I']
    colors = ['r', 'g', 'b']

    for version, params in data.groupby('version'):
        i = order.index(version)
        ax = axes[i]

        for i, (m, df) in enumerate(params.groupby('model')):
            B = np.asarray(df['B'].dropna())
            ax.hist(
                B, label=m, bins=32, alpha=0.5,
                range=[-0.5, 2], normed=True, color=colors[i])

            x = np.linspace(-0.5, 2.0, 100)
            y = util.kde(x[:, None], B, 0.1)
            ax.plot(x, np.exp(y), color=colors[i])

        ax.set_title(version)
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.9')

    axes[0].legend(loc='best')

    fig.set_figwidth(10)
    fig.set_figheight(3)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
