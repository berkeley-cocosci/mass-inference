#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(results_path, fig_paths):

    sns.set_style("white", {'axes.edgecolor': util.lightgrey})
    cols = ['lower', 'median', 'upper']

    fall_responses = pd\
        .read_csv(results_path.joinpath("fall_responses.csv"))\
        .groupby(['version', 'block'])\
        .get_group(('GH', 'B'))\
        .replace({'species': {'model': 'ipe'}})\
        .set_index(['species', 'kappa0', 'stimulus'])[cols]\
        .sortlevel()
    fall_responses['query'] = 'fall'

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_responses_by_stimulus.csv'))\
        .groupby('version')\
        .get_group('H')
    mass_responses = mass_responses\
        .set_index(['species', 'kappa0', 'stimulus'])[cols]\
        .sortlevel()
    mass_responses['query'] = 'mass'

    responses = pd.concat([fall_responses, mass_responses]).reset_index()
    responses['kappa0'] = responses['kappa0'].astype(str)
    responses = responses\
        .set_index(['species', 'kappa0', 'stimulus', 'query'])\
        .unstack(['species', 'query'])\
        .reorder_levels((1, 2, 0), axis=1)\
        .ix[['-1.0', '1.0']]

    fall_corrs = pd\
        .read_csv(results_path.joinpath("fall_response_corrs.csv"))\
        .set_index(['block', 'X', 'Y'])\
        .ix[('B', 'ModelS', 'Human')]

    mass_corrs = pd\
        .read_csv(results_path.joinpath(
            "mass_responses_by_stimulus_corrs.csv"))\
        .set_index(['version', 'X', 'Y'])

    pearson = r"$r={median:.2f}$, $95\%\ \mathrm{{CI}}\ [{lower:.2f},\ {upper:.2f}]$"
    corrs = []
    corrs.append(pearson.format(**dict(fall_corrs)))
    corrs.append(pearson.format(**dict(
        mass_corrs.ix[('H', 'IPE', 'Human')])))
    corrs.append(pearson.format(**dict(
        mass_corrs.ix[('H', 'Empirical', 'Human')])))

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot([0, 1], [0, 1], '--', color=util.lightgrey, linewidth=2, zorder=1)
    for ax in (ax2, ax3):
        ax.plot([0, 1], [0.5, 0.5], '-', color=util.lightgrey, linewidth=2, zorder=1)
        ax.plot([0.5, 0.5], [0, 1], '-', color=util.lightgrey, linewidth=2, zorder=1)


    colors = {
        '-1.0': util.colors[0],
        '1.0': util.colors[2]
    }

    for kappa0, df in responses.groupby(level='kappa0'):
        empirical = df['empirical']
        ipe = df['ipe']
        human = df['human']

        label = r"$\kappa=%.1f$" % 10 ** float(kappa0)

        # left subplot (fall responses)
        x = ipe['fall', 'median']
        x_lerr = x - ipe['fall', 'lower']
        x_uerr = ipe['fall', 'upper'] - x
        y = human['fall', 'median']
        y_lerr = y - human['fall', 'lower']
        y_uerr = human['fall', 'upper'] - y

        ax1.errorbar(
            x, y,
            xerr=[x_lerr, x_uerr],
            yerr=[y_lerr, y_uerr],
            marker='o', color=colors[kappa0], ms=6, ls='',
            ecolor=util.darkgrey,
            label=label)

        # middle subplot (ipe mass responses)
        x = ipe['mass', 'median']
        x_lerr = ipe['mass', 'median'] - ipe['mass', 'lower']
        x_uerr = ipe['mass', 'upper'] - ipe['mass', 'median']
        y = human['mass', 'median']
        y_lerr = human['mass', 'median'] - human['mass', 'lower']
        y_uerr = human['mass', 'upper'] - human['mass', 'median']

        ax2.errorbar(x, y, xerr=[x_lerr, x_uerr],
                     yerr=[y_lerr, y_uerr],
                     marker='o', linestyle='', ms=6,
                     color=colors[kappa0], ecolor=util.darkgrey,
                     label=label)

        # right subplot (empirical ipe mass responses)
        x = empirical['mass', 'median']
        x_lerr = empirical['mass', 'median'] - empirical['mass', 'lower']
        x_uerr = empirical['mass', 'upper'] - empirical['mass', 'median']
        y = human['mass', 'median']
        y_lerr = human['mass', 'median'] - human['mass', 'lower']
        y_uerr = human['mass', 'upper'] - human['mass', 'median']

        ax3.errorbar(x, y, xerr=[x_lerr, x_uerr],
                     yerr=[y_lerr, y_uerr],
                     marker='o', linestyle='', ms=6,
                     color=colors[kappa0], ecolor=util.darkgrey,
                     label=label)

    for corr, ax in zip(corrs, (ax1, ax2, ax3)):
        ax.text(xmax - 0.01, ymin + 0.025, corr,
                horizontalalignment='right', fontsize=10)

    ax1.set_xlabel("IPE model, $p(F_t|S_t)$")
    ax1.set_ylabel("Normalized human judgments")
    ax1.set_title("Exp 1+2: Will it fall?")
    ax2.set_xlabel("IPE model, $p(\kappa=10|F_t,S_t)$")
    ax2.set_ylabel("% participants choosing $\kappa=10$")
    ax2.set_title("Exp 1: Which is heavier? (IPE)")
    ax3.set_xlabel("Empirical model, $p(\kappa=10|F_t,S_t)$")
    ax3.set_ylabel("% participants choosing $\kappa=10$")
    ax3.set_title("Exp 1: Which is heavier? (Empirical)")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1])
        if ax == ax1:
            ax.set_yticklabels(["0.0", "0.25", "0.50", "0.75", "1.0"])
        else:
            ax.set_yticklabels(["0", "25", "50", "75", "100"])

    ax3.legend(loc='upper left', fontsize=10, title="True mass ratio")

    fig.set_figheight(3.5)
    fig.set_figwidth(12)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
