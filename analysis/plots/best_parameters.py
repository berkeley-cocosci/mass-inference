#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import pandas as pd
import util
import numpy as np


def plot(results_path, fig_paths):
    mass_err = pd.read_csv(
        results_path.joinpath("mass_accuracy_best_parameters.csv"))
    mass_err = mass_err\
        .set_index(['sigma', 'phi', 'kappa0', 'stimulus'])['sqerr']\
        .sortlevel()

    fall_err = pd.read_csv(
        results_path.joinpath("fall_responses_best_parameters.csv"))
    fall_err = fall_err\
        .set_index(['sigma', 'phi', 'kappa', 'stimulus'])['sqerr']\
        .sortlevel()

    err = pd\
        .concat([mass_err, fall_err])\
        .groupby(level=['sigma', 'phi'])\
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
    ax.set_title("Average MSE")

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
