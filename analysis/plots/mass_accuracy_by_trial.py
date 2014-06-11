#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_accuracy_by_trial.csv'))\
        .groupby(['version', 'class', 'species'])\
        .get_group(('G', 'chance', 'human'))

    fig, ax = plt.subplots()
    x = mass_responses['trial']
    y = mass_responses['median']
    yl = mass_responses['lower']
    yu = mass_responses['upper']

    ax.fill_between(x, yl, yu, alpha=0.3, color='k')
    ax.plot(x, y, color='k', lw=2)
    ax.set_xlim(1, 20)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Fraction correct")

    fig.set_figwidth(6)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_by_trial.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
