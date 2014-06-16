#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    responses = pd\
        .read_csv(results_path.joinpath('mass_responses_by_stimulus.csv'))\
        .groupby('version')\
        .get_group('H')
    responses = responses.ix[responses['stimulus'] != 'prior']

    colors = {
        -1.0: 'r',
        1.0: 'b'
    }

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    for kappa0, df in responses.groupby('kappa0'):
        model = df.groupby('species').get_group('ipe')
        human = df.groupby('species').get_group('human')

        x = model['median']
        xl = x - model['lower']
        xu = model['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax1.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

        model = df.groupby('species').get_group('empirical')
        human = df.groupby('species').get_group('human')

        x = model['median']
        xl = x - model['lower']
        xu = model['upper'] - x

        y = human['median']
        yl = y - human['lower']
        yu = human['upper'] - y

        ax2.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

        model1 = df.groupby('species').get_group('empirical')
        model2 = df.groupby('species').get_group('ipe')

        x = model1['median']
        xl = x - model1['lower']
        xu = model1['upper'] - x

        y = model2['median']
        yl = y - model2['lower']
        yu = model2['upper'] - y

        ax3.errorbar(x, y, xerr=[xl, xu], yerr=[yl, yu],
                     marker='o', linestyle='',
                     color=colors[kappa0], ecolor='k',
                     label="kappa=%s" % kappa0)

    ax1.set_title("IPE vs. Human")
    ax2.set_title("Empirical IPE vs. Human")
    ax3.set_title("Empirical IPE vs. IPE")

    for ax in (ax1, ax2, ax3):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax1.set_xlabel("IPE")
    ax1.set_ylabel("Human")
    ax2.set_xlabel("Empirical IPE")
    ax2.set_ylabel("Human")
    ax3.set_xlabel("Empirical IPE")
    ax3.set_ylabel("IPE")

    fig.set_figwidth(10)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_responses_by_stimulus.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
