#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(results_path, fig_paths):

    version, block = fig_paths[:2]
    fig_paths = fig_paths[2:]

    results = pd.read_csv(results_path.joinpath("fall_responses.csv"))
    groups = results.set_index(['stimulus', 'kappa0'])\
                    .groupby(['version', 'block'])\
                    .get_group((version, block))\
                    .groupby('species')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    colors = {
        -1.0: 'r',
        1.0: 'b'
    }

    x = groups.get_group('model')['median']
    x_lerr = (x - groups.get_group('model')['lower'])\
        .unstack('kappa0')
    x_uerr = (groups.get_group('model')['upper'] - x)\
        .unstack('kappa0')
    x = x.unstack('kappa0')

    y = groups.get_group('human')['median']
    y_lerr = (y - groups.get_group('human')['lower'])\
        .unstack('kappa0')
    y_uerr = (groups.get_group('human')['upper'] - y)\
        .unstack('kappa0')
    y = y.unstack('kappa0')

    ax1.errorbar(
        y[-1.0], y[1.0],
        xerr=[y_lerr[-1.0], y_uerr[-1.0]],
        yerr=[y_lerr[1.0], y_uerr[1.0]],
        marker='o', color='k', ls='', ms=8)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(r"Human ($r_0=0.1$)")
    ax1.set_ylabel(r"Human ($r_0=10.0$)")

    ax2.plot([x[-1.0], x[1.0]], [y[-1.0], y[1.0]], 'k-')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.plot([x[0.0], x[0.0]], [y[-1.0], y[1.0]], 'k-')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)

    for kappa0 in (-1.0, 1.0):
        ax2.errorbar(
            x[kappa0], y[kappa0],
            xerr=[x_lerr[kappa0], x_uerr[kappa0]],
            yerr=[y_lerr[kappa0], y_uerr[kappa0]],
            marker='o', color=colors[kappa0], ms=8, ls='',
            label=r"$r_0=%.1f$" % 10 ** kappa0)
        ax3.errorbar(
            x[0.0], y[kappa0],
            xerr=[x_lerr[kappa0], x_uerr[kappa0]],
            yerr=[y_lerr[kappa0], y_uerr[kappa0]],
            marker='o', color=colors[kappa0], ms=8, ls='',
            label=r"$r_0=%.1f$" % 10 ** kappa0)

    ax2.set_xlabel("Mass-sensitive IPE")
    ax3.set_xlabel("Mass-insensitive IPE")

    ax3.legend(loc='lower right', fontsize=11, frameon=False)

    for ax in (ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Human")

    for ax in (ax1, ax2, ax3):
        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.9')

    fig.set_figheight(3.5)
    fig.set_figwidth(12)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
