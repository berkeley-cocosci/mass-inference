#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot_class(cls, results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_responses.csv'))\
        .groupby('class')\
        .get_group(cls)

    colors = {
        'human': 'k',
        'empirical': 'r',
        'ipe': 'b'
    }

    fig, ax = plt.subplots()
    for species, sdf in mass_responses.groupby('species'):
        x = sdf['trial']
        y = sdf['median']
        yl = sdf['lower']
        yu = sdf['upper']

        ax.fill_between(x, yl, yu, alpha=0.3, color=colors[species])
        ax.plot(x, y, color=colors[species], lw=2, label=species)

    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Fraction correct")

    ax.legend(loc='best')
    fig.set_figwidth(6)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_with_model.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


def plot(results_path, fig_path):
    pths = []
    responses = pd.read_csv(results_path.joinpath('mass_responses.csv'))
    lhtypes = list(responses['likelihood'].unique())
    for lhtype in lhtypes:
        pths.extend(plot_lhtype(lhtype, results_path, fig_path))
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
