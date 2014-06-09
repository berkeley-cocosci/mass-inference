#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot_block(block, results_path, fig_path, version):

    results = pd.read_csv(results_path.joinpath(version, "fall_responses.csv"))
    groups = results.set_index(['stimulus', 'kappa0'])\
                    .groupby(['block', 'species'])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

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

    ax1.errorbar(
        y[-1.0], y[1.0],
        xerr=[y_lerr[-1.0], y_uerr[-1.0]],
        yerr=[y_lerr[1.0], y_uerr[1.0]],
        marker='o', color='k', ls='', ms=8)
    ax1.set_xlim(1, 7)
    ax1.set_ylim(1, 7)
    ax1.set_xlabel(r"Human ($r_0=0.1$)")
    ax1.set_ylabel(r"Human ($r_0=10.0$)")

    ax2.plot([x[-1.0], x[1.0]], [y[-1.0], y[1.0]], 'k-')
    ax3.plot([x[0.0], x[0.0]], [y[-1.0], y[1.0]], 'k-')

    for kappa0 in (-1.0, 1.0):
        ax2.errorbar(
            x[kappa0], y[kappa0],
            yerr=[y_lerr[kappa0], y_uerr[kappa0]],
            marker='o', color=colors[kappa0], ms=8, ls='',
            label=r"$r_0=%.1f$" % 10 ** kappa0)
        ax3.errorbar(
            x[0.0], y[kappa0],
            yerr=[y_lerr[kappa0], y_uerr[kappa0]],
            marker='o', color=colors[kappa0], ms=8, ls='',
            label=r"$r_0=%.1f$" % 10 ** kappa0)

    ax2.set_xlabel("Mass-sensitive IPE")
    ax3.set_xlabel("Mass-insensitive IPE")

    ax3.legend(title='Mass ratio', loc='best')

    for ax in (ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 7)
        ax.set_ylabel("Human")

    for ax in (ax1, ax2, ax3):
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.95')

    fig.set_figheight(3.5)
    fig.set_figwidth(12)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath(version, "fall_responses_%s.%s" % (block, ext))
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
