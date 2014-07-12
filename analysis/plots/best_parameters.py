#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_paths):
    mass_corrs = pd.read_csv(
        results_path.joinpath("mass_accuracy_best_parameters.csv"))
    mass_corrs = mass_corrs\
        .set_index(['sigma', 'phi'])\
        .unstack('sigma')

    fall_corrs = pd.read_csv(
        results_path.joinpath("fall_responses_best_parameters.csv"))
    fall_corrs = fall_corrs\
        .set_index(['sigma', 'phi'])\
        .unstack('sigma')

    corrs = (mass_corrs + fall_corrs) / 2.0

    fig, ax = plt.subplots()
    cax = ax.imshow(
        corrs, cmap='gray', interpolation='nearest', vmin=0.2, vmax=0.6,
        origin='lower')
    ax.set_xlabel(r"Perceptual uncertainty ($\sigma$)")
    ax.set_ylabel(r"Force uncertainty ($\phi$)")
    ax.set_xticks(range(len(corrs.columns))[::2])
    ax.set_yticks(range(len(corrs.index))[::2])
    ax.set_xticklabels(corrs['pearsonr'].columns[::2])
    ax.set_yticklabels(corrs.index[::2])
    ax.set_title("Average correlations")

    fig.colorbar(cax, ticks=[0.2, 0.3, 0.4, 0.5, 0.6])

    fig.set_figwidth(5)
    fig.set_figheight(4)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
