import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import pandas as pd
import scipy.stats
import os
import yaml

import model_observer as mo
from stats_tools import normalize

# global variables
HUMAN_DATA_PATH = "../data/human"
OLD_HUMAN_DATA_PATH = "../data/old-human"
MODEL_DATA_PATH = "../data/model"


######################################################################
# Data handling
######################################################################


def parse_condition(cond):
    """Condition ids are formatted as:

    obstype-group-fbtype[-ratio[-cb]]

    Where:

      obstype: observer type, human (H) or model (M)
      group: experimental condition, e.g. 'E'
      fbtype: feedback type, can be 'nfb' (no feedback), 'fb' (text
              feedback), 'vfb' (video feedback)
      ratio (optional): mass ratio, e.g. '10'
      cb (optional): counterbalanced, can be 0 or 1

    """
    args = cond.split("-")
    obstype = args[0]
    group = args[1]
    fbtype = args[2]
    ratio = args[3] if len(args) > 3 else None
    cb = args[4] if len(args) > 4 else None
    return obstype, group, fbtype, ratio, cb


def get_bad_pids(conds, thresh=1):
    """Get the pids from participants who either repeated the
    experiment (invalid pids) or failed the posttest (failed pids).

    Parameters
    ----------
    conds: list of condition ids to check posttests for
    thresh: number of wrong answers allowed on the posttest

    Returns
    -------
    out: tuple of (failed pids, invalid pids)

    """

    # load valid pids
    valid_pids = np.load(os.path.join(HUMAN_DATA_PATH, "valid_pids.npy"))
    valid_pids = [("%03d" % int(x[0]), str(x[1])) for x in valid_pids]

    # load model data
    d = load_model('stability-sameheight')
    rawipe, ipe_samps, rawtruth, feedback, kappas, sstim = d

    # load posttest data
    failed_pids = []
    invalid_pids = []
    for cond in conds:
        df = load_turk_df(cond, "posttest")
        hstim = zip(*df.columns)[1]

        # get the correct stimuli indices
        idx = np.nonzero(np.array(hstim)[:, None] == np.array(sstim)[None])[1]

        # model stability
        ofb = feedback[0, idx]

        wrong = (df != ofb).sum(axis=1)
        bad = wrong > thresh
        bad_pids = list((bad).index[(bad).nonzero()])

        failed_pids.extend(bad_pids)
        invalid_pids.extend(
            [x for x in bad.index if (x, cond) not in valid_pids])

    return sorted(set(failed_pids)), sorted(set(invalid_pids))


def load_turk_df(conditions, mode="experiment", istim=True,
                 itrial=True, exclude=None):
    """Load human data from mechanical turk into a pandas DataFrame
    format.

    Parameters
    ----------
    conditions: list of experimental conditions to load
    mode (default "experiment"): the type of data to load, can be
        "experiment", "training", "posttest", or "queries"
    istim (default True): whether to include stimuli names in column
        labels
    itrial (default True): whether to include trial numbers in column
        labels
    exclude (default None): a list of pids to exclude from the data

    Returns
    -------
    df: pandas DataFrame with pids for rows and stimuli/trial for cols

    """

    # argument parsing
    assert istim or itrial
    if not hasattr(conditions, "__iter__"):
        conditions = [conditions]
    if exclude is None:
        exclude = []

    dfs = []
    for condition in conditions:
        # load data file
        path = os.path.join(
            HUMAN_DATA_PATH,
            "consolidated_data",
            "%s_data~%s.npz" % (mode, condition))
        datafile = np.load(path)

        # extract response data
        data = datafile['data']['response']
        # extract stimuli
        stims = []
        for s in datafile['stims']:
            if s.endswith("_cb-0") or s.endswith("_cb-1"):
                stims.append(s[:len(s)-len("_cb-0")])
            else:
                stims.append(s)
        # extract trial numbers
        trial = datafile['data']['trial'][:, 0]
        # extract participant ids
        pids = datafile['pids']
        datafile.close()

        # exclude bad pids
        mask = np.array([p not in exclude for p in pids])
        pids = [x for x in pids if x not in exclude]
        # make dataframe column labels
        columns = []
        if itrial:
            columns.append(trial)
        if istim:
            columns.append(stims)

        # create pandas DataFrame
        df = pd.DataFrame(data.T[mask], columns=columns, index=pids)
        dfs.append(df)

    # concatenate all the dataframes
    df = pd.concat(dfs)
    colnames = []
    # set column and index names
    if itrial:
        colnames.append("trial")
    if istim:
        colnames.append("stimulus")
    df.columns.names = colnames
    df.index.names = ["pid"]

    return df


def load_turk(conds, suffixes, thresh=1):
    """Load human mechanical turk data for given conditions and
    suffixes. Excludes invalid and failed pids. Returns a tuple of
    dictionaries corresponding to each part of the experiment; each
    dictionary contains DataFrame objects.

    Parameters
    ----------
    conds: list of conditions to load, e.g. 'C-vfb-10'
    suffixes: list of suffixes to load, e.g. '-cb0'

    Returns
    -------
    training: dictionary of training data
    posttest: dictionary of posttest data
    experiment: dictionary of posttest data
    queries: dictionary of explicit mass query data

    """

    training = {}
    posttest = {}
    experiment = {}
    queries = {}

    allconds = [c+s for c in conds for s in suffixes]

    # get bad pids
    failed_pids, invalid_pids = get_bad_pids(allconds, thresh=thresh)
    print "%d failed posttest" % len(failed_pids)
    print "%d ineligible" % len(invalid_pids)
    pids = sorted(set(failed_pids + invalid_pids))

    for cond in conds:
        # load training data
        training["H-"+cond] = pd.concat([
            load_turk_df(
                cond+s,
                "training",
                exclude=pids,
                istim=True,
                itrial=True)
            for s in suffixes])

        # load posttest data
        posttest["H-"+cond] = pd.concat([
            load_turk_df(
                cond+s,
                "posttest",
                exclude=pids,
                istim=True,
                itrial=True)
            for s in suffixes])

        # load experiment data
        experiment["H-"+cond] = pd.concat([
            load_turk_df(
                cond+s,
                "experiment",
                exclude=pids,
                istim=True,
                itrial=True)
            for s in suffixes])

        # load explicit mass query data (if feedback condition)
        if cond.split("-")[1] != "nfb":
            queries["H-"+cond] = pd.concat([
                load_turk_df(
                    cond+s,
                    "queries",
                    exclude=pids,
                    istim=False,
                    itrial=True)
                for s in suffixes])

    return training, posttest, experiment, queries


def load_old_human(predicate):
    """Load human data from original experiments.

    Returns
    -------
    (human means,
     raw human data,
     stimuli names)

    """

    with np.load(os.path.join(OLD_HUMAN_DATA_PATH, predicate + ".npz")) as fh:
        human = fh['human']
        rawhuman = fh['rawhuman']
        stims = fh['stims']

    return human, rawhuman, stims


def load_model(predicate, nthresh0=0, nthresh=0.4, fstim=None):
    """Load IPE simulation data.

    Parameters
    ----------
    predicate: type of data to load, e.g. 'stability', 'direction', 'all'
    nthresh0: more than this fraction of blocks need to have moved for
        the tower to have fallen in "truth" samples
    nthresh: more than this fraction of blocks need to have moved for
        the tower to have fallen in IPE samples
    fstim: list of stimuli to filter by

    Returns
    -------
    (raw IPE summary data,
     IPE samples,
     raw truth summary data,
     truth data,
     kappas)

    """

    # load the summary model data
    with np.load(os.path.join(MODEL_DATA_PATH, predicate + ".npz")) as fh:
        stims0 = fh['stims']

        # figure out the filter indices
        if fstim is not None:
            sstim = np.array(stims0)
            idx = np.nonzero((sstim[:, None] == fstim[None, :]))[0]
            stims = None
            assert idx.size > 0
        else:
            idx = slice(None)
            stims = stims0

        truth = fh['truth'][:, idx]
        ipe = fh['ipe'][idx]
        kappas = fh['kappas']

    # compute feedback/truth
    if hasattr(nthresh0, "__iter__"):
        lower, upper = nthresh0
        print "Note: interval %s specified for nthresh0" % (nthresh0,)
        feedback = (truth > upper).astype('f8')
        feedback[truth <= upper] = np.nan
        feedback[truth <= lower] = 0
        feedback[np.isnan(truth)] = np.nan
    else:
        feedback = (truth > nthresh0).astype('f8')
        feedback[np.isnan(truth)] = 0.5

    # compute ipe samples
    ipe_samps = (ipe > nthresh).astype('f8')
    ipe_samps[np.isnan(ipe)] = 0.5

    out = (ipe,
           ipe_samps,
           truth,
           feedback,
           kappas)
    if stims is not None:
        out = out + (stims,)
    return out


def generate_model_responses(conds, experiment, queries, feedback, ipe_samps,
                             kappas, ratios, n_trial, n_fake=2000):
    """Generate model stability responses, explicit mass judgments,
    and belief over time for each experimental condition in 'conds'.

    """

    rso = np.random.RandomState(1)

    model_belief = {}
    model_experiment = {}
    model_queries = {}
    n_kappas = len(kappas)

    for cond in conds:
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        if obstype == "M" or fbtype == "vfb":
            continue

        cols = experiment[cond].columns
        order = np.argsort(zip(*cols)[0])
        undo_order = np.argsort(order)

        ## Stability judgments
        if fbtype == 'nfb':
            fb = np.empty((3, n_trial)) * np.nan
            prior = np.zeros((3, n_kappas,))
            # uniform
            prior[0, :] = 1
            prior = normalize(np.log(prior), axis=1)[1]
        else:
            ridx = ratios.index(float(ratio))
            fb = feedback[:, order][ridx]
            prior = None
        responses, model_theta = mo.simulateResponses(
            n_fake, fb, ipe_samps[order], kappas,
            prior=prior, p_ignore=0.0, smooth=True, rso=rso)

        ## Model belief over time
        if fbtype == "nfb":
            newcond = "M-%s-%s-10" % (group, fbtype)
            model_experiment[newcond] = pd.DataFrame(
                responses[:, 0][:, undo_order],
                columns=cols)
            model_belief[newcond] = model_theta[0]
        else:
            newcond = "M-%s-%s-%s" % (group, fbtype, ratio)
            model_experiment[newcond] = pd.DataFrame(
                responses[:, undo_order],
                columns=cols)
            model_belief[newcond] = model_theta

        ## Explicit mass judgments
        if fbtype == "fb":
            cols = queries[cond].columns
            idx = np.array(cols) - np.arange(len(cols)) - 7
            theta = np.exp(model_theta[idx])
            r1 = ratios.index(1.0)
            p1 = theta[:, r1]
            if float(ratio) > 1:
                pcorrect = np.sum(theta[:, r1:], axis=1) + (p1 / 2.)
                other = 0.1
            else:
                pcorrect = np.sum(theta[:, :r1], axis=1) + (p1 / 2.)
                other = 10
            r = rso.rand(n_fake, 1) < pcorrect[None]
            responses = np.empty(r.shape)
            responses[r] = float(ratio)
            responses[~r] = other
            model_queries[newcond] = pd.DataFrame(
                responses, columns=cols)

    return model_experiment, model_queries, model_belief


######################################################################
# Plotting functions
######################################################################


def make_cmap(name, c1, c2, c3):
    """Make a cmap that fades from color c1 to color c3 through c2"""
    colors = {
        'red': (
            (0.0, c1[0], c1[0]),
            (0.50, c2[0], c2[0]),
            (1.0, c3[0], c3[0]),),
        'green': (
            (0.0, c1[1], c1[1]),
            (0.50, c2[1], c2[1]),
            (1.0, c3[1], c3[1]),),
        'blue': (
            (0.0, c1[2], c1[2]),
            (0.50, c2[2], c2[2]),
            (1.0, c3[2], c3[2]))}
    cmap = matplotlib.colors.LinearSegmentedColormap(name, colors, 1024)
    return cmap


def savefig(path, fignum=None, close=True, width=None, height=None, ext=None, verbose=False):
    """Save a figure from pyplot"""

    if fignum is None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum)

    if ext is None:
        ext = ['']

    if width:
        fig.set_figwidth(width)
    if height:
        fig.set_figheight(height)

    directory = os.path.split(path)[0]
    filename = os.path.split(path)[1]
    if directory == '':
        directory = '.'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for ex in ext:
        if ex == '':
            name = filename
        else:
            name = filename + "." + ex

        if verbose:
            print "Saving figure to %s...'" % (
                os.path.join(directory, name)),
        plt.savefig(os.path.join(directory, name))
        if verbose:
            print "Done"

    if close:
        plt.clf()
        plt.cla()
        plt.close()


def plot_theta(nrow, ncol, idx, theta, title,
               exp=2.718281828459045, cmap='hot',
               fontsize=12, vmin=0, vmax=1):
    try:
        idx.cla()
    except AttributeError:
        ax = plt.subplot(nrow, ncol, idx)
    else:
        ax = idx
    ax.cla()
    if exp is not None:
        x = exp ** np.log(theta.T)
    else:
        x = theta.T
    img = ax.imshow(
        x,
        aspect='auto', interpolation='nearest',
        vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_ylabel("Mass Ratio", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    return img


def plot_smoothing(samps, stims, istim, kappas, legend=True):
    n_trial, n_kappas, n_samples = samps.shape
    alpha = np.sum(samps, axis=-1) + 0.5
    beta = np.sum(1-samps, axis=-1) + 0.5
    pfell_mean = alpha / (alpha + beta)
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    y_mean = mo.IPE(np.ones(n_trial), samps, kappas, smooth=True)

    colors = cm.hsv(np.round(np.linspace(0, 220, len(istim))).astype('i8'))
    xticks = np.linspace(-1.3, 1.3, 7)
    xticks10 = 10 ** xticks
    xticks10[xticks < 0] = np.round(xticks10[xticks < 0], decimals=2)
    xticks10[xticks >= 0] = np.round(xticks10[xticks >= 0], decimals=1)
    yticks = np.linspace(0, 1, 3)

    plt.ylim(0, 1)
    plt.xticks(xticks, xticks10)
    plt.xlabel("Mass ratio ($r$)", fontsize=14)
    plt.yticks(yticks, yticks)
    plt.ylabel("P(fall|$r$, $S$)", fontsize=14)
    plt.grid(True)
    order = istim

    for idx in xrange(len(istim)):
        i = order[idx]
        x = kappas
        plt.plot(x, y_mean[i],
                 color=colors[idx],
                 linewidth=3)
        plt.errorbar(x, pfell_mean[i], pfell_std[i], None,
                     color=colors[idx], fmt='o',
                     markeredgecolor=colors[idx],
                     markersize=5,
                     label=str(stims[i]).split("_")[1])
    if legend:
        plt.legend(loc=8, prop={'size': 12}, numpoints=1,
                   scatterpoints=1, ncol=3, title="Stimuli")


def plot_belief(fignum, r, c, model_beliefs, kappas, cmap, cond_labels):
    ratios = np.round(10 ** np.array(kappas), decimals=2)
    n = r*c
    exp = np.e

    fig = plt.figure(fignum)
    plt.clf()
    gs = gridspec.GridSpec(r, c+1, width_ratios=[1]*c + [0.1])

    plt.suptitle(
        "Ideal observer posterior beliefs",
        fontsize=16)
    plt.subplots_adjust(
        wspace=0.2,
        hspace=0.3,
        left=0.1,
        right=0.93,
        top=0.8,
        bottom=0.1)

    vmax = 0
    for cond in sorted(model_beliefs.keys()):
        v = np.exp(model_beliefs[cond]).max()
        if v > vmax:
            vmax = v

    for i, cond in enumerate(sorted(model_beliefs.keys())):
        irow, icol = np.unravel_index(i, (r, c))
        ax = plt.subplot(gs[irow, icol])
        n_trial = model_beliefs[cond].shape[1] - 1
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        subjname = "$r_0=%s$, order %s" % (ratio, group)
        img = plot_theta(
            None, None, ax,
            np.exp(model_beliefs[cond]),
            subjname,
            exp=exp,
            cmap=cmap,
            fontsize=14,
            vmin=0,
            vmax=vmax)

        yticks = np.array([kappas.index(-1.0),
                           kappas.index(0.0),
                           kappas.index(1.0)])
        if (i % c) == 0:
            plt.yticks(yticks, ratios[yticks], fontsize=14)
            plt.ylabel("Mass ratio ($r$)", fontsize=14)
        else:
            plt.yticks(yticks, [])
            plt.ylabel("")

        xticks = np.linspace(0, n_trial, 4).astype('i8')
        if (n-i) <= c:
            plt.xticks(xticks, xticks, fontsize=14)
            plt.xlabel("Trial number ($t$)", fontsize=14)
        else:
            plt.xticks(xticks, [])

    cticks = np.linspace(0, vmax, 5)
    cax = plt.subplot(gs[:, -1])
    cb = fig.colorbar(img, ax=ax, cax=cax, ticks=cticks)
    cb.set_ticklabels(np.round(cticks, decimals=2))
    cax.set_title("$\Pr(r|B_t)$", fontsize=14)


######################################################################
# Model/analysis helper functions
######################################################################

def random_model_lh(responses, n_trial, t0=None, tn=None):
    if t0 is None:
        t0 = 0
    if tn is None:
        tn = n_trial

    lh = {}
    for cond in responses.keys():
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        lh[cond] = np.zeros((responses[cond].shape[0], 1))
        lh[cond] += np.log(0.5) * (tn - t0)

    return lh


def block_lh(responses, feedback, ipe_samps, prior, kappas,
             t0=None, tn=None, f_smooth=True, p_ignore=0.0):

    n_trial = ipe_samps.shape[0]
    if t0 is None:
        t0 = 0
    if tn is None:
        tn = n_trial

    lh = {}
    for cond in responses.keys():
        order = np.argsort(zip(*responses[cond].columns)[0])
        order = order[t0:tn]

        # trial-by-trial likelihoods of judgments
        resp = np.asarray(responses[cond])[..., order]
        lh[cond] = mo.EvaluateObserver(
            resp, feedback[..., order], ipe_samps[order],
            kappas, prior=prior, p_ignore=p_ignore, smooth=f_smooth)

    return lh


def CI(data, conds):
    # consolidate data from different orderings
    newdata = {}
    for cond in conds:
        if cond not in data:
            continue
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        key = (obstype, fbtype, ratio)
        if key not in newdata:
            newdata[key] = []
        newdata[key].append(data[cond])
    for key in newdata.keys():
        obstype, fbtype, ratio = key
        newcond = "%s-all-%s-%s" % (obstype, fbtype, ratio)
        data[newcond] = np.vstack(newdata[key])

    # now compute statistics
    stats = {}
    bmv = scipy.stats.bayes_mvs
    for cond in conds:
        if cond not in data:
            continue
        shape = data[cond].shape
        assert len(shape) == 2
        info = []
        for i in xrange(shape[1]):
            d = data[cond][:, i]
            shape = d.shape
            if (d == d[0]).all():
                mean = lower = upper = d[0]
            else:
                mean, (lower, upper) = bmv(d, alpha=0.95)[0]
            assert not np.isnan(mean), d
            sum = np.sum(d)
            n = d.size
            info.append([mean, lower, upper, sum, n])
        if len(info) == 1:
            stats[cond] = np.array(info[0])
        else:
            stats[cond] = np.array(info)

    return stats


# def infer_beliefs(data, ipe_samps, feedback, kappas, f_smooth):
#     post = {}
#     n_trial, n_kappas, n_samp = ipe_samps.shape
#     # prior = np.zeros(n_kappas) + 0.5
#     # prior[kappas.index(0.0)] = 1
#     # prior = np.abs(kappas.index(0.0) - np.arange(n_kappas)) + 10
#     # prior[kappas.index(0.0)] += 10
#     # prior = 1. / prior
#     # prior = np.zeros(n_kappas) + 0.5
#     # prior[kappas.index(0.2)] = 1
#     # prior[kappas.index(-0.2)] = 1
#     prior = np.ones(n_kappas)
#     prior = normalize(np.log(prior))[1]
#     # lhF = np.log(mo.IPE(feedback, ipe_samps, kappas, f_smooth))
#     print prior
#     for cond in data:
#         obstype, group, fbtype, ratio, cb = parse_condition(cond)
#         d = data[cond]
#         order = np.argsort(zip(*d.columns)[0])
#         logpr = np.empty((d.shape[0], n_trial+1, n_kappas))
#         logpr[:, 0] = prior.copy()
#         lhJ = np.log(mo.IPE(np.asarray(d)[:, order], ipe_samps[order], kappas, f_smooth))
#         # if fbtype == "nfb":
#         lh = lhJ.copy()
#         # else:
#         #     lh = lhJ + lhF[kappas.index(np.log10(float(ratio)))][order]
#         for t in xrange(0, n_trial):
#             logpr[:, t+1] = normalize(lh[:, t] + logpr[:, t], axis=1)[1]
#         post[cond] = np.exp(logpr)
#     return post


# def infer_CI(data, conds):
#     # consolidate data from different orderings
#     newdata = {}
#     for cond in conds:
#         if cond not in data:
#             continue
#         obstype, group, fbtype, ratio, cb = parse_condition(cond)
#         key = (obstype, fbtype, ratio)
#         if key not in newdata:
#             newdata[key] = []
#         newdata[key].append(data[cond])
#     for key in newdata.keys():
#         obstype, fbtype, ratio = key
#         newcond = "%s-all-%s-%s" % (obstype, fbtype, ratio)
#         data[newcond] = np.vstack(newdata[key])
#     return data


def model_lhs(data, feedback, nofeedback, ipe_samps,
              kappas, ratios, conds,
              f_smooth=True, p_ignore=0.0):

    ir1 = list(kappas).index(0.0)
    ir10 = list(kappas).index(1.0)
    ir01 = list(kappas).index(-1.0)

    n_trial, n_kappas, n_samp = ipe_samps.shape

    # random model
    model_random = CI(random_model_lh(data, n_trial), conds)

    # fixed models
    theta = np.zeros((1, n_kappas))
    theta[:, ir1] = 1
    theta = normalize(np.log(theta), axis=1)[1]
    fb = np.empty((1, n_trial)) * np.nan
    model_true1 = CI(block_lh(
        data, fb, ipe_samps, theta, kappas,
        f_smooth=f_smooth, p_ignore=p_ignore), conds)

    # theta = np.zeros((4, n_kappas))
    # theta[0, :ir1] = 1
    # theta[1, ir1+1:] = 1
    # theta[2, :ir1] = 1
    # theta[3, ir1+1:] = 1
    # # theta[0, ir01] = 1
    # # theta[1, ir10] = 1
    # theta = normalize(np.log(theta), axis=1)[1]
    # fb = np.empty((4, n_trial)) * np.nan
    # model_fixed = CI(block_lh(
    #     data, fb, ipe_samps, theta, kappas,
    #     f_smooth=f_smooth, p_ignore=p_ignore), conds)

    # model_true = {}
    # model_wrong = {}
    # for cond in model_fixed:
    #     obstype, group, fbtype, ratio, cb = parse_condition(cond)
    #     if ratio != '0':
    #         if float(ratio) > 1:
    #             model_true[cond] = model_fixed[cond][1].copy()
    #             model_wrong[cond] = model_fixed[cond][2]
    #         else:
    #             model_true[cond] = model_fixed[cond][0].copy()
    #             model_wrong[cond] = model_fixed[cond][3].copy()

    model_uniform = CI(block_lh(
        data, nofeedback, ipe_samps, None, kappas,
        f_smooth=f_smooth, p_ignore=p_ignore), conds)

    # learning models
    theta = np.ones((2, n_kappas))
    theta = normalize(np.log(theta), axis=1)[1]
    fb = feedback[[ir01, ir10]]
    model_not_fixed = CI(block_lh(
        data, fb, ipe_samps, theta, kappas,
        f_smooth=f_smooth, p_ignore=p_ignore), conds)

    model_learn = {}
    for cond in model_not_fixed:
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        if ratio != '0':
            if float(ratio) > 1:
                model_learn[cond] = model_not_fixed[cond][1].copy()
            else:
                model_learn[cond] = model_not_fixed[cond][0].copy()

    # all the models
    mnames = np.array([
        "Random",
        "Equal",
        # "Uniform Wrong",
        "Uniform",
        # "Uniform Correct",
        "Learning",
	])

    models = [
        model_random,
        model_true1,
        # model_wrong,
        model_uniform,
        # model_true,
        model_learn,
    ]

    return models, mnames


def make_performance_table(groups, conds, models, mnames):
    performance_table = []
    table_conds = []
    table_models = list(mnames.copy())
    table_models.remove("Random")
    table_models.remove("Equal")
    print table_models
    for i, group in enumerate(groups):
        performance_table.append([])
        table_conds.append([])
        for cidx, cond in enumerate(conds):
            obstype, grp, fbtype, ratio, cb = parse_condition(cond)
            if (grp != group) or obstype == "M":
                continue
            data = []
            for x in xrange(len(models)):
                if mnames[x] not in table_models:
                    continue
                if cond in models[x]:
                    data.append(models[x][cond].copy())
            if len(data) == 0:
                continue
            data = np.array(data)
            table_conds[-1].append(cond)
            performance_table[-1].append(data)
    performance_table = np.array(performance_table)
    print performance_table.shape
    #norm = normalize(performance_table[..., 3], axis=-1)[1]
    norm = performance_table[..., 3] / performance_table[..., 4]
    performance_table[..., 3] = norm
    return performance_table, table_conds, table_models


def print_performance_table(performance_table, table_conds,
                            groups, mnames, cond_labels):
    for gidx, group in enumerate(groups):
        if group == "C":
            name = "Random"
        elif group == "E":
            name = "Diagnostic"
        else:
            continue

        # group, condition, model, stat
        table = performance_table[gidx, :, :, 3]
        samplesize = performance_table[gidx, :, :, 4]
        print r"\subfloat{\begin{tabular}[b]{|c|c||%s|}\hline" % ("c"*len(mnames))
        print (r"\textbf{%s} & $n$ & " % name) + " & ".join(mnames) + r"\\\hline"
        for lidx, line in enumerate(table):
            midx = np.nanargmax(np.array(line, dtype='f8'))
            cond = table_conds[gidx][lidx]
            obstype, group, fbtype, ratio, cb = parse_condition(cond)
            if fbtype == "fb" and ratio == "0.1":
                clabel = r"\fblow{}"
            elif fbtype == "fb" and ratio == "10":
                clabel = r"\fbhigh{}"
            elif fbtype == "vfb" and ratio == "0.1":
                clabel = r"\vfblow{}"
            elif fbtype == "vfb" and ratio == "10":
                clabel = r"\vfbhigh{}"
            elif fbtype == "nfb":
                clabel = r"\nfb{}"

            entries = [str(clabel), str(int(samplesize[lidx][0]))]
            entries.extend([
                "" if np.isnan(line[idx]) else
                (r"\textbf{%.2f}" if idx == midx
                 else "%.2f") % float(line[idx])
                for idx in xrange(len(line))])
            print " & ".join(entries) + r"\\"
        print r"\hline"
        print r"\end{tabular}} &"


def plot_model_comparison(info, chance=True, static=True, learning=True):
    basekwargs = {
        'align': 'center',
        'width': 1,
    }

    colors = {
        'Static': 'b',
        'Learning': 'g'
    }

    table_models = info['table_models']
    heights = info['heights']
    chance_val = info['chance']
    orders = info['orders']
    fbratios = info['fbratios']
    fbtypes = info['fbtypes']

    best = np.nanargmax(heights, axis=-1)
    best[0, 0, 0] = 2
    x = [0, 1, 2.5, 3.5, 5, 6, 7.5, 8.5]*2

    fig, axes = plt.subplots(1, 2)

    if static or learning:
        for i in xrange(heights.size):
            oidx, ridx, fidx, midx = np.unravel_index(i, heights.shape)

            if not static and midx == 0:
                continue
            if not learning and midx == 1:
                continue

            order = orders[oidx]
            ratio = fbratios[ridx]
            fbtype = fbtypes[fidx]
            model = table_models[midx]

            height = heights[oidx, ridx, fidx, midx]

            kwargs = basekwargs.copy()
            kwargs.update({
                'color': colors[model]
            })

            if oidx == 0 and ridx == 0 and fidx == 0:
                kwargs['label'] = model

            axes[oidx].bar(x[i], height, **kwargs)

            if best[oidx, ridx, fidx] == midx:
                axes[oidx].text(
                    x[i], height, '*',
                    fontweight='bold', fontsize=16,
                    ha='center', color='w')

    for ax in axes:
        ax.set_ylim(-30.1, -24.8)
        xlim = [-1, 9.5]
        ax.set_xlim(*xlim)

        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.xaxis.tick_top()

        ax.set_xticks([0.5, 3, 5.5, 8])
        ax.set_xticklabels(['Binary', 'Video']*2, fontsize=12)

        rkwargs = {
            'horizontalalignment': 'center',
            'fontsize': 22
        }
        if not (static or learning):
            rkwargs['color'] = 'w'

        ax.text(1.75, -24.25, '$r=0.1$', **rkwargs)
        ax.text(6.75, -24.25, '$r=10$', **rkwargs)

        if chance:
            ax.plot(xlim, [chance_val]*2, 'k--', linewidth=2, label='Chance')

    ax1, ax2 = axes
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    ax1.set_yticklabels([])
    ax1.set_ylabel("Average log likelihood", fontsize=16)
    ax1.set_xlabel("Order 1", fontsize=22)
    ax2.set_xlabel("Order 2", fontsize=22)

    if learning:
        ax1.legend(loc='lower left', ncol=4, fontsize=12, frameon=False,
                   bbox_to_anchor=(0, -0.04))
    elif static or chance:
        ax1.legend(loc='lower left', ncol=4, fontsize=12, frameon=False,
                   bbox_to_anchor=(0, -0.03))

    ax2.text(9.25, -30, '* = best model fit',
             horizontalalignment='right',
             fontsize=12)

    fig.set_figwidth(11)
    fig.set_figheight(4)

    plt.subplots_adjust(wspace=0.02, right=0.95, top=0.8)


def wcih(queries, meta):
    orders = meta['orders']
    conds = meta['conds']

    emjs = {}
    for cond in conds:
        if cond not in queries:
            continue
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        emjs[cond] = np.asarray(queries[cond]) == float(ratio)
        index = np.array(queries[cond].columns, dtype='i8')
        X = np.array(index)-6-np.arange(len(index))-1

    emj_stats = CI(emjs, conds)

    nratio = len(meta['fbratios'])
    norder = len(meta['orders'])
    nfb = len(meta['fbtypes'])
    ntrial = X.size
    shape = (nratio, norder, nfb, ntrial)

    # which color is heavier
    wcih = {
        'mean': np.empty(shape),
        'upper': np.empty(shape),
        'lower': np.empty(shape),
        'sig': np.empty(shape, dtype='bool'),
        'sum': np.empty(shape),
        'n': np.empty(shape),
    }

    for i, group in enumerate(orders):
        for cidx, cond in enumerate(conds):
            if cond not in emj_stats:
                continue
            obstype, grp, fbtype, ratio, cb = parse_condition(cond)
            if (grp != group):
                continue
            if obstype == "M":
                fbtype = 'model'
            mean, lower, upper, sums, n = emj_stats[cond].T
            f = lambda x: scipy.stats.binom_test(x, n[0], 0.5)
            binom = [f(x) for x in sums]

            ridx = meta['fbratios'].index(ratio)
            gidx = meta['orders'].index(grp)
            fidx = meta['fbtypes'].index(fbtype)
            idx = (ridx, gidx, fidx)

            wcih['mean'][idx] = mean
            wcih['lower'][idx] = lower
            wcih['upper'][idx] = upper
            wcih['sig'][idx] = np.array(binom) < 0.05
            wcih['sum'][idx] = sums
            wcih['n'][idx] = n[0]

    return X, wcih


def load_opts(filename):
    with open(filename) as fh:
        opts = fh.read()
    return yaml.load(opts)
