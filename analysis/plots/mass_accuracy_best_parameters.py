#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_paths):
    corrs = pd.read_csv(
        results_path.joinpath("mass_accuracy_best_parameters.csv"))
    corrs = corrs\
        .set_index(['sigma', 'phi'])\
        .unstack('sigma')

    fig, ax = plt.subplots()
    cax = ax.imshow(
        corrs, cmap='gray', interpolation='nearest', vmin=-0.3, vmax=0.3,
        origin='lower')
    ax.set_xlabel(r"Perceptual uncertainty ($\sigma$)")
    ax.set_ylabel(r"Force uncertainty ($\phi$)")
    ax.set_xticks(range(len(corrs.columns))[::2])
    ax.set_yticks(range(len(corrs.index))[::2])
    ax.set_xticklabels(corrs['pearsonr'].columns[::2])
    ax.set_yticklabels(corrs.index[::2])
    ax.set_title("Correlations for \"which is heavier?\"")

    fig.colorbar(cax, ticks=[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.3])

    fig.set_figwidth(5)
    fig.set_figheight(4)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
