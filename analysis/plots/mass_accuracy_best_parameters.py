#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):
    corrs = pd.read_csv(
        results_path.joinpath("mass_accuracy_best_parameters.csv"))
    corrs = corrs\
        .set_index(['sigma', 'phi'])\
        .unstack('phi')

    fig, ax = plt.subplots()
    cax = ax.imshow(
        corrs, cmap='gray', interpolation='nearest', vmin=-0.2, vmax=0.3)
    ax.set_xlabel(r"Force uncertainty ($\phi$)")
    ax.set_ylabel(r"Perceptual uncertainty ($\sigma$)")
    ax.set_xticks(range(len(corrs.columns)))
    ax.set_yticks(range(len(corrs.index)))
    ax.set_xticklabels(corrs['pearsonr'].columns)
    ax.set_yticklabels(corrs.index)

    fig.colorbar(cax, ticks=[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3])

    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_best_parameters.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
