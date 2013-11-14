import itertools
from itertools import izip
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


def smoothing_before():
    fig, ax = plt.subplots(1, 1)
    # customize x axis
    ax.set_xticks([-1.0, -0.7, -0.3, 0, 0.3, 0.7, 1.0])
    ax.set_xticklabels([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax.set_xlabel(r"Mass ratio ($r$)")
    # customize y axis
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"P(fall | $r$, $S$)")
    # overall plot customization
    ax.grid(True)
    ax.set_title("IPE likelihood smoothing")
    return fig, ax


def smoothing_after(fig, ax):
    # legend
    ax.legend(loc='lower center')
    plt.tight_layout()


def belief_before(fb_belief):
    orders = fb_belief.order.unique()
    ratios = fb_belief.ratio.unique()

    fig = plt.figure()
    grid = ImageGrid(
        fig, 111,
        nrows_ncols=(len(orders), len(ratios)),
        axes_pad=0.4,
        share_all=True,
        cbar_mode='single',
        label_mode='L',
        cbar_pad=0.25)

    kappas = list(fb_belief.kappa.unique())
    yticks = [kappas.index(x) for x in [-1, 0, 1]]
    yticklabels = ['0.1', '1.0', '10']

    for ax in grid:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Mass ratio ($r$)")

    return fig, grid


def belief_after(fig, grid, im):
    grid.cbar_axes[0].colorbar(im)
    grid.cbar_axes[0].set_title(r"$p(r)$")


def learning_curves_before(mass_stats):
    orders = mass_stats.order.unique()
    ratios = mass_stats.ratio.unique()
    trials = mass_stats.trial.unique()

    plotnames = itertools.product(orders, ratios)
    ncurves = len(orders)*len(ratios)
    fig, axes = plt.subplots(
        1, ncurves, sharex=True, sharey=True)
    fig.set_figwidth(16)
    fig.set_figheight(4)

    axes_dict = {}
    axes[0].set_ylabel("Proportion correct")

    for ax, (order, ratio) in izip(axes.flat, plotnames):
        ax.set_xticks(trials)
        ax.set_xlim(trials.min(), trials.max())
        ax.set_xlabel("Trial")
        ax.set_yticks([0.25, 0.50, 0.75, 1.0])
        ax.set_ylim(0.25, 1)
        ax.set_title(r"$r_0=%s$, order %s" % (ratio, order))
        ax.plot([1, 40], [0.5, 0.5], 'k:')
        axes_dict[(order, ratio)] = ax

    return fig, axes_dict


def learning_curves_after(fig, axes):
    fig.axes[-1].legend(loc='lower right')
    plt.tight_layout()


def model_comparison_before(lh, sig):
    conds = lh.reset_index('model').index.unique()

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(12)

    ax.set_xlim(0, len(conds))
    ax.set_xticks(np.arange(len(conds)) + 0.5)
    ax.set_xticklabels(map(str, conds))

    yticks = np.arange(-30, -23)
    ax.set_ylim(-30.5, -23.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(map(str, yticks))
    ax.set_ylabel(r"Log-likelihood (better model fit $\rightarrow$)")

    def label_diff(x0, x1, y, text):
        x = (x0 + x1) / 2.

        props = {'connectionstyle': 'bar',
                 'arrowstyle': '-',
                 'shrinkA': 15,
                 'shrinkB': 15,
                 'lw': 1}

        ax.annotate(
            text,
            xy=(x, y + 0.5),
            zorder=10,
            ha='center',
            fontsize=12)
        ax.annotate(
            '',
            xy=(x0, y + 0.1),
            xytext=(x1, y + 0.1),
            arrowprops=props,
            zorder=10)

    lh_groups = lh.groupby(level=['order', 'ratio', 'feedback'])
    sig_groups = sig.groupby(level=['order', 'ratio', 'feedback'])
    for i, cond in enumerate(conds):
        lh_cond = lh_groups.get_group(cond)
        p = float(sig_groups.get_group(cond))
        if p < 0.05:
            text = "p < 0.05"
            star = "**"
        elif p < 0.01:
            text = "p < 0.01"
            star = "**"
        elif p < 0.1:
            text = "p < 0.1"
            star = "*"
        else:
            text = "p = %.2f" % p
            star = ""
        t = text + "\n" + star
        y = lh_cond['upper'].max()
        label_diff(i + 1./6, i + 5./6, y, t)

    return fig, ax


def model_comparison_after(fig, ax):
    ax.legend(loc='lower right', fontsize=20, frameon=False)
    plt.tight_layout()
