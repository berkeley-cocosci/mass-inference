#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(results_path, fig_paths):
    err = pd.read_csv(
        results_path.joinpath("mass_accuracy_best_parameters.csv"))
    err = err\
        .groupby(['sigma', 'phi'])['sqerr']\
        .mean()\
        .unstack('sigma')

    fig, ax = plt.subplots()
    cax = ax.imshow(
        err, cmap='gray', interpolation='nearest',
        origin='lower')
    ax.set_xlabel(r"Perceptual uncertainty ($\sigma$)")
    ax.set_ylabel(r"Force uncertainty ($\phi$)")
    ax.set_xticks(range(len(err.columns))[::2])
    ax.set_yticks(range(len(err.index))[::2])
    ax.set_xticklabels(err.columns[::2])
    ax.set_yticklabels(err.index[::2])
    ax.set_title("MSE for \"which is heavier?\"")
    ax.grid('off')

    err_arr = np.asarray(err)
    best_y, best_x = np.nonzero(err_arr == err_arr.min())
    best_x = int(best_x)
    best_y = int(best_y)

    rect = plt.Rectangle(
        (best_x - 0.5, best_y - 0.5), 1, 1,
        ec='r', lw=2, fc=None, fill=False)
    ax.add_patch(rect)

    fig.colorbar(cax)

    fig.set_figwidth(5)
    fig.set_figheight(4)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
