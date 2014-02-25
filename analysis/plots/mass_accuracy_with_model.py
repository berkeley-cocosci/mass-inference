#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    mass_responses = pd\
        .read_csv(results_path.joinpath('mass_responses.csv'))\
        .groupby('species')\
        .get_group('human')

    ranks = pd.read_csv(results_path.joinpath('participant_fits.csv'))
    best = ranks.set_index('pid')['rank_1']
    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))
    belief = belief\
        .set_index(['pid', 'likelihood'])\
        .groupby(lambda x: best[x[0]] == x[1])\
        .get_group(True)\
        .reset_index()\
        .pivot_table(
            rows=['kappa0', 'pid'],
            cols='trial',
            values='p')\
        .groupby(level='kappa0')\
        .mean()\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'p'})

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    for i, (kappa0, df) in enumerate(mass_responses.groupby('kappa0')):
        x = df['trial']
        y = df['median']
        yl = df['lower']
        yu = df['upper']

        ax = axes[i]
        ax.fill_between(x, yl, yu, alpha=0.3, color='k')
        ax.plot(x, y, color='k', lw=2, label='human')
        ax.set_title(r"$\kappa_0=%.1f$" % kappa0)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Fraction correct")

    for i, (kappa0, df) in enumerate(belief.groupby('kappa0')):
        axes[i].plot(
            df['trial'],
            df['p'],
            'r-', lw=3,
            label='model')

    axes[1].legend(loc='best')
    fig.set_figwidth(8)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_with_model.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
