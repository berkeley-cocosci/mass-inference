#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot_model(model, results_path, fig_path):

    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))
    belief = belief.groupby('likelihood').get_group('ipe')
    if model == 'best':
        ranks = pd.read_csv(results_path.joinpath('participant_fits.csv'))
        best = ranks.set_index('pid').groupby('rank')['model'].get_group(0)
        belief = belief\
            .set_index(['pid', 'model'])\
            .groupby(lambda x: best[x[0]] == x[1])\
            .get_group(True)\
            .reset_index()

    else:
        belief = belief.groupby('model').get_group(model)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    kax = {
        -1.0: ax1,
        1.0: ax2
    }

    for (kappa0, pid), df in belief.groupby(['kappa0', 'pid']):
        if kappa0 == -1:
            ax = ax1
        else:
            ax = ax2
        ax.plot(df['trial'], df['p'], 'k-', alpha=0.3)

    agg = belief.pivot_table(
        rows=['kappa0', 'pid'],
        cols='trial',
        values='p')
    agg = agg\
        .groupby(level='kappa0')\
        .mean()\
        .stack()\
        .reset_index()\
        .rename(columns={0: 'p'})

    for kappa0, ax in kax.iteritems():
        ix = agg['kappa0'] == kappa0
        ax.plot(
            agg['trial'][ix],
            agg['p'][ix],
            'b-', lw=3)
        ax.set_title(r"$\kappa_0=%.1f$" % kappa0)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1)

    fig.set_figwidth(8)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("model_belief_%s.%s" % (model, ext))
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


def plot(results_path, fig_path):
    pths = []
    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))
    models = list(belief['model'].unique()) + ['best']
    for model in models:
        pths.extend(plot_model(model, results_path, fig_path))
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
