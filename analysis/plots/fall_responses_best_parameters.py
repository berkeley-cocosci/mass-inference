#!/usr/bin/env python

"""Plots a grid of correlations for each pair of sigma/phi parameters. The pair
with the highest correlation is given a red outline.

"""

__depends__ = ["fall_responses_best_parameters.csv"]

import util
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(dest, results_path):
    err = pd.read_csv(
        os.path.join(results_path, "fall_responses_best_parameters.csv"))

    query = util.get_query()
    err = err\
        .groupby('query')\
        .get_group(query)\
        .groupby(['sigma', 'phi'])['pearsonr']\
        .mean()\
        .unstack('sigma')\
        .fillna(0)

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
    ax.set_title("Correlations for \"will it fall?\"")
    ax.grid('off')

    err_arr = np.asarray(err)
    best_y, best_x = np.nonzero(err_arr == err_arr.max())
    best_x = int(best_x)
    best_y = int(best_y)

    rect = plt.Rectangle(
        (best_x - 0.5, best_y - 0.5), 1, 1,
        ec='r', lw=2, fc=None, fill=False)
    ax.add_patch(rect)

    fig.colorbar(cax)

    fig.set_figwidth(6.5)
    fig.set_figheight(5.2)
    plt.draw()
    plt.tight_layout()

    for pth in dest:
        util.save(pth, close=False)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    plot(args.to, args.results_path)
