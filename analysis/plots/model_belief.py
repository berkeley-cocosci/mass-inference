#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    belief = pd.read_csv(results_path.joinpath('model_belief_agg.csv'))

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

    for kappa0, ax in kax.iteritems():
        ax.set_title(r"$\kappa_0=%.1f$" % kappa0)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 1)

    fig.set_figwidth(8)
    plt.tight_layout()

    pths = [fig_path.joinpath("model_belief.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
