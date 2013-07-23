import matplotlib.pyplot as plt
import analysis_tools as at
import numpy as np
from snippets import graphing
import itertools
import matplotlib.gridspec as gridspec


def join(*args):
    return " ".join(args)


def _wcih_trial(ax, trial, ridx, all_trials, wcih, opts):

    mean = wcih['mean'][ridx, :, :-1]
    sig = wcih['sig'][ridx]

    fbratios = opts['fbratios']
    orders = opts['orders']
    fbtypes = opts['fbtypes']
    colors = opts['colors']
    lightcolors = opts['lightcolors']
    hatches = opts['hatches']
    labels = opts['labels']

    xlim = -0.75, 4.25
    ylim = 0.3, 1.01
    ratio = fbratios[ridx]
    bar_x = [0, 1, 2.5, 3.5]
    xticks = [x+0.5 for x in bar_x[::2]]

    ax.plot(xlim, [0.5, 0.5], 'k:', linewidth=3)

    basekwargs = {'align': 'center'}

    tidx = all_trials.index(trial)
    for i in xrange(mean[..., 0].size):
        oidx, fidx = np.unravel_index(i, mean.shape[:-1])
        idx = (oidx, fidx, tidx)
        order = orders[oidx]
        fbtype = fbtypes[fidx]

        kwargs = basekwargs.copy()
        kwargs.update({
            'color': colors.get(fbtype, None),
            'label': labels.get(join(order, fbtype, ratio), None),
            'align': 'center',
            'hatch': hatches.get(order, None),
        })

        if kwargs['hatch']:
            kwargs['edgecolor'] = lightcolors[kwargs['color']]

        ax.bar(bar_x[i], mean[idx], **kwargs)

        if kwargs['hatch']:
            ax.bar(bar_x[i], mean[idx], color='none', **basekwargs)

        if not sig[idx]:
            ax.text(bar_x[i], mean[idx]+0.01, 'ns',
                    horizontalalignment='center')

    # set axis limits
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)

    graphing.clear_top(ax)
    graphing.clear_right(ax)

    # set axis tick locations and labels
    ax.set_xticks(xticks)
    ax.set_xticklabels([])
    ax.set_yticks([])

    # disable tick marks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')


def _wcih_all(ax, ridx, oidx, all_trials, wcih, opts):

    mean = wcih['mean'][ridx, oidx]
    lower = wcih['lower'][ridx, oidx]
    upper = wcih['upper'][ridx, oidx]

    orders = opts['orders']
    fbtypes = opts['fbtypes']
    colors = opts['colors']
    linestyles = opts['linestyles']
    labels = opts['labels']

    for i in xrange(mean[..., 0].size):
        fidx, = np.unravel_index(i, mean.shape[:-1])
        idx = (fidx,)
        order = orders[oidx]
        fbtype = fbtypes[fidx]

        basekwargs = {
            'color': colors.get(fbtype, None),
        }
        kwargs = basekwargs.copy()
        kwargs.update({
            'label': labels.get(join(fbtype), None),
            'linestyle': linestyles.get(join(order, fbtype)),
            'linewidth': 5,
        })

        ax.fill_between(all_trials, lower[idx], upper[idx],
                        alpha=0.2, **basekwargs)
        ax.plot(all_trials, mean[idx], **kwargs)

    xlim = 1, 40
    ylim = 0.3, 1.01
    xticks = all_trials

    ax.plot(xlim, [0.5, 0.5], 'k:', linewidth=3)

    # set axis limits
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)

    # set axis tick locations and labels
    ax.set_yticks([])
    ax.set_xticks(xticks)
    ax.xaxis.tick_top()
    ax.set_xticklabels([])

    # disable tick marks
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('none')


def plot_wcih(all_trials, wcih, opts):
    fbratios = opts['fbratios']
    orders = opts['orders']
    trials = [1, 8]
    labels = opts['labels']
    colors = opts['colors']

    nratio = len(fbratios)
    norder = len(orders)
    ntrial = len(trials)

    fig = plt.figure()
    gs = gridspec.GridSpec(
        nratio, ntrial + norder,
        width_ratios=[1]*ntrial + [1.25]*norder)
    axes = np.array([[plt.subplot(gs[i, j])
                      for j in range(ntrial + norder)]
                    for i in range(nratio)])

    items = itertools.product(
        range(nratio),
        range(ntrial + norder))

    for i, idx in enumerate(items):
        ridx = idx[0]
        if idx[1] >= ntrial:
            tidx = None
            trial = None
            oidx = idx[1] - ntrial
            order = orders[oidx]
        else:
            tidx = idx[1]
            trial = trials[tidx]
            oidx = None
            order = None

        ax = axes[idx]
        if tidx is not None:
            _wcih_trial(ax, trial, ridx, all_trials, wcih, opts)

            if ridx == 0:
                ax.set_title("Trial %d" % trial, fontsize=28,
                             position=(0.5, 1.15))
            if tidx == 0:
                ratio = fbratios[ridx]
                ax.set_ylabel("Percent correct")
                ax.text(-2.65, 0.78, r"$r_0=%s$" % ratio,
                        rotation=90, ha='center',
                        fontsize=28)
                ax.set_yticks(np.arange(0.3, 1.1, 0.1))
                ax.set_yticklabels(["%d%%" % x for x in xrange(30, 110, 10)])
                graphing.set_ylabel_coords(-0.2, ax=ax)

            if tidx == 1:
                graphing.clear_left(ax)

            if ridx == 1:
                xticklabels = [
                    "%s" % labels.get(join(order)).capitalize()
                    for order in orders]
                ax.set_xticklabels(xticklabels, fontsize=20)

        else:
            _wcih_all(ax, ridx, oidx, all_trials, wcih, opts)

            if ridx == 0:
                ax.set_title("All Trials", fontsize=28, position=(0.5, 1.15))
                xticklabels = map(str, map(int, all_trials))
                ax.set_xticklabels(xticklabels, fontsize=16)
            # if oidx == 0:
            #     ax.set_yticks(np.arange(0.3, 1.1, 0.1))
            #     ax.set_yticklabels(["%d%%" % x for x in xrange(30, 110, 10)])
            if ridx == 1:
                ax.set_xlabel(labels[order].capitalize())

    ax = axes[0, 1]
    legend_lines = [
        plt.Line2D((0, 1), (0, 0), color=colors.get('fb'), lw=10),
        plt.Line2D((0, 1), (0, 0), color=colors.get('vfb'), lw=10),
        plt.Line2D((0, 1), (0, 0), color='k', lw=10)
    ]
    ax.legend(legend_lines,
              ['Binary f.b.', 'Visual f.b.', 'Learning IPE'],
              loc='upper left', frameon=False, ncol=3,
              bbox_to_anchor=(-1.05, 1.183), fontsize=16)

    fig.set_figwidth(16)
    fig.set_figheight(7.5)

    plt.subplots_adjust(left=0.09, bottom=0.05, top=0.875,
                        right=0.99, wspace=0.1, hspace=0.2)


def model_comparison(heights, chance_val, opts):
    basekwargs = {
        'align': 'center',
        'width': 1,
    }

    colors = {
        'Fixed Belief IPE': 'm',
        'Learning IPE': '#FFa500'
    }

    table_models = opts['table_models']
    table_conds = opts['table_conds']
    orders = opts['orders']

    best = np.nanargmax(heights, axis=-1)
    best[0, 0] = 2
    x = [0, 1, 2.5, 3.5, 5.5, 6.5, 8, 9, 11, 12]*2

    fig, axes = plt.subplots(1, 2)

    for i in xrange(heights.size):
        oidx, cidx, midx = np.unravel_index(i, heights.shape)
        cond = table_conds[oidx][cidx]
        obstype, grp, fbtype, ratio, cb = at.parse_condition(cond)
        model = table_models[midx]
        height = heights[oidx, cidx, midx]

        kwargs = basekwargs.copy()
        kwargs.update({
            'color': colors[model]
        })

        if oidx == 1 and cidx == 0:
            kwargs['label'] = model

        axes[oidx].bar(x[i], height, **kwargs)

        if best[oidx, cidx] == midx:
            axes[oidx].text(
                x[i], height, '*',
                fontweight='bold',
                ha='center', color='w')

    for i, ax in enumerate(axes):
        ax.set_ylim(-30.1, -24.8)
        xlim = [-1, 13]
        ax.set_xlim(*xlim)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.xaxis.tick_top()

        ax.set_xticks([0.5, 3, 6, 8.5, 11.5])
        xticklabels = ['Binary', 'Visual']*2
        xticklabels += ['No f.b.']
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("%s order" % opts['labels'][orders[i]].capitalize(),
                      fontsize=24)

        rkwargs = {
            'horizontalalignment': 'center',
        }

        ax.text(1.75, -24.25, r'$r_0=0.1$', **rkwargs)
        ax.text(7.25, -24.25, r'$r_0=10$', **rkwargs)

        ax.plot(xlim, [chance_val]*2, 'k--', linewidth=2, label='Chance')

        graphing.clear_bottom(ax)
        graphing.clear_right(ax)
        graphing.clear_left(ax)
        ax.yaxis.set_ticks_position('none')

    ax1, ax2 = axes
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax1.set_yticklabels([])
    ax1.set_ylabel(r"Better model fit $\rightarrow$")

    ax2.legend(loc='lower left', frameon=False,
               bbox_to_anchor=(0, -0.04))

    ax2.text(12.75, -29.85, '* = best model fit',
             horizontalalignment='right')

    graphing.clear_left(ax2)

    fig.set_figwidth(16)
    fig.set_figheight(5.15)

    plt.subplots_adjust(wspace=0.1, left=0.03, right=0.99, top=0.75,
                        bottom=0.075)
    plt.suptitle('Average log likelihood of human "will it fall?" judgments',
                 fontsize=28)
