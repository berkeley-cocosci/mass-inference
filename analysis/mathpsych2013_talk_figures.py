import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import analysis_tools as at
import numpy as np
from snippets import graphing
import itertools
from stats_tools import normalize

FACTOR = 1.75
HEIGHT = 7
WIDTH = FACTOR * HEIGHT


def join(*args):
    return " ".join(args)


def wcih_trial(trial, all_trials, wcih, opts):

    fbratios = opts['fbratios']
    orders = opts['orders']
    fbtypes = opts['fbtypes']
    colors = opts['colors']
    lightcolors = opts['lightcolors']
    hatches = opts['hatches']
    labels = opts['labels']

    fig, axes = plt.subplots(1, len(fbratios))

    for ridx, ax in enumerate(axes):
        mean = wcih['mean'][ridx, :, :-1]
        sig = wcih['sig'][ridx]

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

            if sig[idx] < 0.05:
                ptext = "*"
            else:
                ptext = ""

            ax.text(bar_x[i], mean[idx]+0.01, ptext,
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

        if ridx == 0:
            ax.set_ylabel("Percent correct")
            ax.set_yticks(np.arange(0.3, 1.1, 0.1))
            ax.set_yticklabels(["%d%%" % x for x in xrange(30, 110, 10)])
        else:
            graphing.clear_left(ax)

        xticklabels = [
            "%s" % labels.get(join(order)).capitalize()
            for order in orders]
        ax.set_xticklabels(xticklabels, fontsize=20)
        ax.set_xlabel(r"$r_0=%s$" % ratio, fontsize=24)

    fig.set_figwidth(WIDTH)
    fig.set_figheight(HEIGHT)

    plt.subplots_adjust(left=0.11, bottom=0.1, top=0.95,
                        right=0.99, wspace=0.1)

    ax = axes[1]
    legend_lines = [
        plt.Line2D((0, 1), (0, 0), color=colors.get('fb'), lw=10),
        plt.Line2D((0, 1), (0, 0), color=colors.get('vfb'), lw=10),
    ]
    ax.legend(legend_lines,
              ['Binary f.b.', 'Visual f.b.'],
              loc='upper left', frameon=False, ncol=2,
              bbox_to_anchor=(-1.1, 1.05), fontsize=24)


def wcih_all(all_trials, wcih, opts):

    fbratios = opts['fbratios']
    orders = opts['orders']
    fbtypes = opts['fbtypes']
    colors = opts['colors']
    linestyles = opts['linestyles']
    labels = opts['labels']

    fig, axes = plt.subplots(len(fbratios), len(orders))

    items = itertools.product(xrange(len(fbratios)), xrange(len(orders)))
    for ridx, oidx in items:

        ax = axes[ridx, oidx]
        ratio = fbratios[ridx]

        mean = wcih['mean'][ridx, oidx]
        lower = wcih['lower'][ridx, oidx]
        upper = wcih['upper'][ridx, oidx]

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
        xticklabels = [str(x) for x in xticks]
        xticklabels[1] = ""

        ax.plot(xlim, [0.5, 0.5], 'k:', linewidth=3)

        # set axis limits
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)

        # set axis tick locations and labels
        ax.set_yticks([])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        # disable tick marks
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('none')

        ax.set_xlabel("%s, $r_0=%s$" % (
            labels[order].capitalize(),
            ratio))
        ax.xaxis.set_label_coords(0.5, -0.125)

        if oidx == 0:
            ax.set_yticks(np.arange(0.3, 1.1, 0.1))
            ax.set_yticklabels(["%d%%" % x for x in xrange(30, 110, 10)])
            ax.set_ylabel("Percent correct")

    ax = axes[1, 1]
    legend_lines = [
        plt.Line2D((0, 1), (0, 0), color=colors.get('fb'), lw=10),
        plt.Line2D((0, 1), (0, 0), color=colors.get('vfb'), lw=10),
        plt.Line2D((0, 1), (0, 0), color='k', lw=10)
    ]
    ax.legend(legend_lines,
              ['Binary f.b.', 'Visual f.b.', 'Learning IPE'],
              loc='lower right', frameon=True, ncol=1, fontsize=20)

    fig.set_figwidth(WIDTH)
    fig.set_figheight(HEIGHT)

    plt.subplots_adjust(left=0.11, bottom=0.1, top=0.975,
                        right=0.975, wspace=0.1, hspace=0.4)


def model_comparison(experiment, queries, X, wcih, nofeedback, feedback,
                     ipe_samps, kappas, opts,
                     m_chance=True, m_fixed=True, m_learning=True):

    nbs = 10000
    fbtypes = ['fb', 'vfb']
    fbratios = ['0.1', '10']
    orders = ['C', 'E']

    basekwargs = {
        'align': 'center',
        'width': 1,
        'ecolor': 'k',
    }

    colors = {
        'Fixed Belief IPE': 'm',
        'Learning IPE': '#FFa500'
    }

    data = []
    keys = []
    i = 0

    n_kappas = ipe_samps.shape[1]

    x = [0, 1, 2.5, 3.5, 5.5, 6.5, 8, 9]*2
    fig, axes = plt.subplots(1, 2)

    items = itertools.product(orders, fbratios, fbtypes)
    for order, fbratio, fbtype in items:

        k = "H-%s-%s-%s" % (order, fbtype, fbratio)
        keys.append(k)
        ki = kappas.index(np.log10(float(fbratio)))

        q = np.asarray(queries[k]) == float(fbratio)
        learned = np.all(q[:, -2:], axis=-1)

        ex = experiment[k][learned]
        n = ex.shape[0]
        ns = n
        if n == 0:
            data.append(np.ones(nbs)*np.nan)
            data.append(np.ones(nbs)*np.nan)
            i += 2
            continue

        rso = np.random.RandomState(2302938)
        idx = rso.randint(0, n, ns*nbs)

        for model in ('Fixed Belief IPE', 'Learning IPE'):
            if model == 'Fixed Belief IPE':
                fb = nofeedback
                prior = None
            elif model == 'Learning IPE':
                fb = feedback[[ki]]
                prior = np.ones((1, n_kappas))
                prior = normalize(np.log(prior), axis=1)[1]
            else:
                raise ValueError(model)

            vals = at.block_lh(
                {k: ex},
                fb, ipe_samps, prior, kappas,
                f_smooth=True, p_ignore=0.0)
            bd = np.mean(vals[k][idx].reshape((ns, nbs)), axis=0)
            data.append(bd)

            m = np.log(0.5) * ex.shape[1]
            chance = m

            # bootstrap standard error (68% CIs)
            lower, median, upper = np.percentile(
                bd, [15.8655254, 50, 84.1344746])
            lerr = median - lower
            uerr = upper - median

            kwargs = basekwargs.copy()
            kwargs.update({
                'color': colors[model]
            })

            fplot = ((model == "Fixed Belief IPE" and m_fixed) or
                     (model == "Learning IPE" and m_learning))
            if fplot:
                oidx = orders.index(order)
                axes[oidx].bar(
                    x[i], median - chance,
                    yerr=np.array([lerr, uerr])[:, None],
                    bottom=chance, **kwargs)

            i += 1

    data = np.array(data)
    dd = data.reshape((data.shape[0]/2, 2, nbs))
    if (m_fixed and m_learning):
        for i in xrange(dd.shape[0]):
            order = keys[i].split("-")[1]
            oidx = orders.index(order)
            p = (dd[i, 0] > dd[i, 1]).mean(axis=-1)
            ptext = "$p"

            if p < 0.05:
                sig = "*"
            else:
                sig = ""

            if p < 0.01:
                ptext += "<0.01$"
            elif p < 0.05:
                ptext += "<0.05$"
            elif p < 0.1:
                ptext += "<0.1$" % p
            else:
                ptext += "=%.2f$" % p

            text = "%s\n%s" % (ptext, sig)

            axes[oidx].text(
                x[i*2]+0.5, -24.25, text,
                fontsize=16, horizontalalignment='center')

    for i, ax in enumerate(axes):
        ylim = -29, -23.5
        yticks = np.arange(ylim[0], ylim[1]+1, 1)
        xlim = [-1, 10]
        if m_chance:
            ax.plot(xlim, [chance, chance], 'k-', lw=4)

        ax.set_xlim(*xlim)
        ax.set_xticks([0.5, 3, 6, 8.5])
        xticklabels = [
            'Binary\n$r=0.1$', 'Visual\n$r=0.1$',
            'Binary\n$r=10$', 'Visual\n$r=10$']
        ax.set_xticklabels(xticklabels)
        ax.xaxis.set_ticks_position('none')
        ax.tick_params(axis='x', labelsize=16)
        order = opts['labels'][orders[i]].capitalize()
        ax.set_title(order, fontsize=24, position=(0.5, 0.95))

        graphing.clear_top(ax)
        graphing.clear_right(ax)

        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        if i == 0:
            ax.set_yticklabels([str(int(y)) for y in yticks])
            ax.yaxis.set_ticks_position('none')
            ax.tick_params(axis='y', labelsize=16)
            ax.set_ylabel(r"Better model fit $\rightarrow$")
        else:
            ax.set_yticklabels([])
            graphing.clear_left(ax)
            ax.yaxis.set_ticks_position('none')

    ax1, ax2 = axes

    legend_lines = [
        plt.Line2D((0, 1), (0, 0),
                   color=colors.get('Fixed Belief IPE'), lw=10),
        plt.Line2D((0, 1), (0, 0),
                   color=colors.get('Learning IPE'), lw=10),
        plt.Line2D((0, 1), (0, 0),
                   color='k', lw=5),
    ]
    axes[1].legend(
        legend_lines,
        ['Fixed', 'Learning', 'Chance'],
        loc='lower center', frameon=False, ncol=3,
        fontsize=20, bbox_to_anchor=(0.3, 0.01))

    axes[0].text(-0.5, -25,
                 r"p-values reflect "
                 "$p(\mathrm{fixed}\ >\ \mathrm{learning})$\n"
                 "error bars are one standard error",
                 fontsize=14, horizontalalignment='left')

    fig.set_figwidth(WIDTH)
    fig.set_figheight(HEIGHT)

    plt.subplots_adjust(
        wspace=0.1, left=0.09, right=0.99,
        top=0.95, bottom=0.11)


def plot_belief(r, c, model_beliefs, kappas, cmap, cond_labels):
    ratios = np.round(10 ** np.array(kappas), decimals=2)
    n = r*c
    exp = np.e

    fig = plt.figure()
    plt.clf()
    gs = gridspec.GridSpec(r, c+1, width_ratios=[1]*c + [0.1])

    plt.subplots_adjust(
        wspace=0.2,
        hspace=0.3,
        left=0.075,
        right=0.95,
        top=0.95,
        bottom=0.1)

    fig.set_figwidth(WIDTH)
    fig.set_figheight(HEIGHT)

    vmax = 0
    for cond in sorted(model_beliefs.keys()):
        v = np.exp(model_beliefs[cond]).max()
        if v > vmax:
            vmax = v

    for i, cond in enumerate(sorted(model_beliefs.keys())):
        irow, icol = np.unravel_index(i, (r, c))
        ax = plt.subplot(gs[irow, icol])
        n_trial = model_beliefs[cond].shape[0] - 1
        obstype, group, fbtype, ratio, cb = at.parse_condition(cond)
        if group == "C":
            order = "order 1"
        elif group == "E":
            order = "order 2"
        else:
            raise ValueError(order)
        subjname = "$r_0=%s$, %s" % (ratio, order)
        img = at.plot_theta(
            None, None, ax,
            np.exp(model_beliefs[cond]),
            subjname,
            exp=exp,
            cmap=cmap,
            fontsize=22,
            vmin=0,
            vmax=vmax)

        yticks = np.array([kappas.index(-1.0),
                           kappas.index(0.0),
                           kappas.index(1.0)])
        if (i % c) == 0:
            plt.yticks(yticks, ratios[yticks])
            plt.ylabel("Mass ratio ($r$)", fontsize=20)
        else:
            plt.yticks(yticks, [])
            plt.ylabel("")

        xticks = np.linspace(0, n_trial, 5).astype('i8')
        if (n-i) <= c:
            plt.xticks(xticks, xticks)
            plt.xlabel("Trial number ($t$)", fontsize=20)
        else:
            plt.xticks(xticks, [])

    cticks = np.linspace(0, vmax-0.01, 5)
    cax = plt.subplot(gs[:, -1])
    cb = fig.colorbar(img, ax=ax, cax=cax, ticks=cticks)
    cb.set_ticklabels(np.round(cticks, decimals=2))
    cax.set_title("$\Pr(r|B_t)$", fontsize=16)
