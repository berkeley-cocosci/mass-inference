#!/usr/bin/env python

import matplotlib.pyplot as plt
import pandas as pd
import util


def plot(results_path, fig_path):

    responses = pd\
        .read_csv(results_path.joinpath('mass_responses.csv'))\
        .groupby('class')
    classes = ['learning', 'static', 'chance', 'best']

    colors = {
        'human': 'k',
        'model': 'r',
    }

    fig, axes = plt.subplots(1, len(classes))
    for i, cls in enumerate(classes):
        df = responses.get_group(cls)
        ax = axes[i]

        for species, sdf in df.groupby('species'):
            if species == 'empirical':
                species = 'model'

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
        ax.set_title(cls.capitalize())

    axes[-1].legend(loc='best')
    fig.set_figwidth(16)
    plt.draw()
    plt.tight_layout()

    pths = [fig_path.joinpath("mass_accuracy_with_model.%s" % ext)
            for ext in ('png', 'pdf')]
    for pth in pths:
        util.save(pth, close=False)
    return pths


if __name__ == "__main__":
    util.make_plot(plot)
