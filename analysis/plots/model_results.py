#!/usr/bin/env python

import util

import sys
import matplotlib.pyplot as plt
import pandas as pd


def plot(results_path, fig_paths):

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

    pearson = r"r={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
    corrs = []
    corrs.append(pearson.format(**dict(fall_corrs)))
    corrs.append(pearson.format(**dict(
        mass_corrs.ix[('H', 'Empirical', 'Human')])))
    corrs.append(pearson.format(**dict(
        mass_corrs.ix[('H', 'IPE', 'Human')])))

    xmin = -0.05
    xmax = 1.05
    ymin = -0.05
    ymax = 1.05

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

    for ax in (ax1, ax2, ax3):
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8)

    colors = {
        '-1.0': 'r',
        '1.0': 'b'
    }

    ax1.plot(
        [responses.ix['-1.0']['ipe', 'fall', 'median'],
         responses.ix['1.0']['ipe', 'fall', 'median']],
        [responses.ix['-1.0']['human', 'fall', 'median'],
         responses.ix['1.0']['human', 'fall', 'median']],
        'k-')

    for kappa0, df in responses.groupby(level='kappa0'):
        empirical = df['empirical']
        ipe = df['ipe']
        human = df['human']

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
            label=r"$r_0=%.1f$" % 10 ** float(kappa0))

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
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

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
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

    for corr, ax in zip(corrs, (ax1, ax2, ax3)):
        ax.text(xmax - 0.01, ymin + 0.025, corr,
                horizontalalignment='right', fontsize=9)

    ax1.set_xlabel("IPE")
    ax1.set_ylabel("Human")
    ax1.set_title("Will it fall? (Exp 1+2a)")
    ax2.set_xlabel("IPE Likelihood")
    ax2.set_title("Which is heavier? (Exp 1)")
    ax3.set_xlabel("Empirical Likelihood")
    ax3.set_title("Which is heavier? (Exp 1)")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        util.clear_right(ax)
        util.clear_top(ax)
        util.outward_ticks(ax)
        ax.set_axis_bgcolor('0.9')

    ax1.legend(loc='upper left', fontsize=11, frameon=False)

    fig.set_figheight(3.5)
    fig.set_figwidth(12)
    plt.draw()
    plt.tight_layout()

    for pth in fig_paths:
        util.save(pth, close=False)


if __name__ == "__main__":
    util.make_plot(plot, sys.argv[1:])
