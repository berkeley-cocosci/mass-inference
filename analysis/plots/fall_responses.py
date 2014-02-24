#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot_block(block, results_path, fig_path):

    results = pd.read_csv(results_path.joinpath("fall_responses.csv"))
    groups = results.set_index(['stimulus', 'kappa0'])\
                    .groupby(['block', 'species'])

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    colors = {
        -1.0: 'r',
        1.0: 'b'
    }

    x = groups.get_group((block, 'model'))['median'].unstack('kappa0')
    y = groups.get_group((block, 'human'))['median']
    y_lerr = (y - groups.get_group((block, 'human'))['lower'])\
        .unstack('kappa0')
    y_uerr = (groups.get_group((block, 'human'))['upper'] - y)\
        .unstack('kappa0')
    y = y.unstack('kappa0')

    for kappa0 in (-1.0, 1.0):
        ax1.errorbar(
            x[kappa0], y[kappa0],
            yerr=[y_lerr[kappa0], y_uerr[kappa0]],
            marker='o', color=colors[kappa0], ms=8, ls='',
            label=r"$r_0=%.1f$" % 10 ** kappa0)
        ax2.errorbar(
            x[0.0], y[kappa0],
            yerr=[y_lerr[kappa0], y_uerr[kappa0]],
            marker='o', color=colors[kappa0], ms=8, ls='',
            label=r"$r_0=%.1f$" % 10 ** kappa0)

    ax1.plot([x[-1.0], x[1.0]], [y[-1.0], y[1.0]], 'k-')
    ax2.plot([x[0.0], x[0.0]], [y[-1.0], y[1.0]], 'k-')

    ax1.legend(title='Mass ratio', loc='best')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 7)
    ax1.set_xlabel("IPE")
    ax2.set_xlabel("IPE")
    ax1.set_ylabel("Human")
    ax1.set_title("Mass-sensitive predictions")
    ax2.set_title("Mass-insensitive predictions")

    fig.set_figwidth(8)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("fall_responses_%s.%s" % (block, ext))
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


def plot(results_path, fig_path):
    pths = []
    for block in ('A', 'B'):
        pths.extend(plot_block(block, results_path, fig_path))
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
