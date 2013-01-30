import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy.stats
import os
import time

#import cogphysics
#import cogphysics.lib.circ as circ
#import cogphysics.lib.nplib as npl
#import cogphysics.lib.rvs as rvs
#import cogphysics.lib.stats as stats
import cogphysics.tower.analysis_tools as tat
import model_observer as mo

from cogphysics.lib.corr import xcorr, partialcorr

from joblib import Memory
memory = Memory(cachedir="cache", mmap_mode='c', verbose=0)

######################################################################
# Data handling

def parse_condition(cond):
    args = cond.split("-")
    obstype = args[0]
    group = args[1]
    fbtype = args[2]
    ratio = args[3] if len(args) > 3 else None
    cb = args[4] if len(args) > 4 else None
    return obstype, group, fbtype, ratio, cb

def get_bad_pids(conds, thresh=1):

    # load valid pids
    valid_pids = np.load("../../turk-experiment/data/valid_pids.npy")
    valid_pids = [("%03d" % int(x[0]), str(x[1])) for x in valid_pids]
    
    # load model data
    rawmodel, sstim, smeta = tat.load_model(1)#0)
    pfell, nfell, fell_persample = tat.process_model_stability(
        rawmodel, mthresh=0.095, zscore=False)
    # convert model stim ids
    from cogphysics import RESOURCE_PATH
    pth = os.path.join(RESOURCE_PATH, 'cpobj_conv_stability.pkl')
    with open(pth, "r") as fh:
        conv = pickle.load(fh)
        sstim = np.array([conv[x] for x in sstim])

    # load posttest data
    all_pids = []
    for cond in conds:
        df = load_turk_df(cond, "posttest")
        hstim = zip(*df.columns)[1]

        # get the correct stimuli indices
        idx = np.nonzero(np.array(hstim)[:, None] == np.array(sstim)[None])[1]

        # model stability 
        ofb = (fell_persample[0,0,0,:,0] > 0)[idx]
            
        wrong = (df != ofb).sum(axis=1)
        bad = wrong > thresh
        bad_pids = list((bad).index[(bad).nonzero()])

        all_pids.extend(bad_pids)
        all_pids.extend([x for x in bad.index if (x, cond) not in valid_pids])

    return sorted(set(all_pids))

# def load_turk(thresh=1, istim=True, itrial=True):
#     training = {}
#     posttest = {}
#     experiment = {}
#     queries = {}

#     suffix = ['-cb0', '-cb1']
#     #conds = ['B-fb-10', 'B-fb-0.1', 'B-nfb-10']
#     conds = ['C-vfb-10', 'C-vfb-0.1', 'C-fb-10', 'C-fb-0.1', 'C-nfb-10',
#              'E-vfb-10', 'E-vfb-0.1', 'E-fb-10', 'E-fb-0.1', 'E-nfb-10']
#     allconds = [c+s for c in conds for s in suffix]
#     pids = get_bad_pids(allconds, thresh=thresh)
#     print "Bad pids (%d): %s" % (len(pids), pids)

#     kwargs = {
#         "exclude": pids,
#         "istim": istim,
#         "itrial": itrial
#         }

#     for cond in allconds:
#         training[cond] = load_turk_df(cond, "training", **kwargs)
#         posttest[cond] = load_turk_df(cond, "posttest", **kwargs)
#         experiment[cond] = load_turk_df(cond, "experiment", **kwargs)
#         if cond.split("-")[1] != "nfb":
#             queries[cond] = load_turk_df(cond, "queries", **kwargs)

#     return training, posttest, experiment, queries

def load_turk(thresh=1):
    training = {}
    posttest = {}
    experiment = {}
    queries = {}

    suffix = ['-cb0', '-cb1']
    #conds = ['B-fb-10', 'B-fb-0.1', 'B-nfb-10']
    conds = ['C-vfb-10', 'C-vfb-0.1', 'C-fb-10', 'C-fb-0.1', 'C-nfb-10',
             'E-vfb-10', 'E-vfb-0.1', 'E-fb-10', 'E-fb-0.1', 'E-nfb-10']
    allconds = [c+s for c in conds for s in suffix]
    pids = get_bad_pids(allconds, thresh=thresh)
    print "Bad pids (%d): %s" % (len(pids), pids)

    # ltraining, lposttest, lexperiment = load_turk_learning(
    #     thresh=thresh, itrial=True)[:3]
    # lqueries = load_turk_learning(
    #     thresh=thresh, istim=False)[3]

    # training = {}
    # posttest = {}
    # experiment = {}
    # queries = {}

    # suffix = ['-cb0', '-cb1']
    # #conds = ['B-fb-10', 'B-fb-0.1', 'B-nfb-10']
    # conds = ['C-vfb-10', 'C-vfb-0.1', 'C-fb-10', 'C-fb-0.1', 'C-nfb-10',
    #          'E-vfb-10', 'E-vfb-0.1', 'E-fb-10', 'E-fb-0.1', 'E-nfb-10']
        
    for cond in conds:
        training["H-"+cond] = pd.concat([
            load_turk_df(cond+s, "training", exclude=pids, istim=True, itrial=True) 
            for s in suffix])
        posttest["H-"+cond] = pd.concat([
            load_turk_df(cond+s, "posttest", exclude=pids, istim=True, itrial=True) 
            for s in suffix])
        experiment["H-"+cond] = pd.concat([
            load_turk_df(cond+s, "experiment", exclude=pids, istim=True, itrial=True) 
            for s in suffix])
        if cond.split("-")[1] != "nfb":
            queries["H-"+cond] = pd.concat([
                load_turk_df(cond+s, "queries", exclude=pids, istim=False, itrial=True) 
                for s in suffix])
        # posttest[cond] = pd.concat([lposttest[cond+s] for s in suffix])
        # experiment[cond] = pd.concat([lexperiment[cond+s] for s in suffix])
        # if cond.split("-")[1] != "nfb":
        #     queries[cond] = pd.concat([lqueries[cond+s] for s in suffix])

    return training, posttest, experiment, queries

        
def load_turk_df(conditions, mode="experiment", istim=True, itrial=True, exclude=None):
    assert istim or itrial
    
    if not hasattr(conditions, "__iter__"):
        conditions = [conditions]
    if exclude is None:
        exclude = []
            
    dfs = []
    for condition in conditions:
        path = "../../turk-experiment/data/consolidated_data/%s_data~%s.npz" % (mode, condition)
        # print "Loading '%s'" % path
        datafile = np.load(path)
        data = datafile['data']['response']
        stims = []
        for s in datafile['stims']:
            if s.endswith("_cb-0") or s.endswith("_cb-1"):
                stims.append(s[:len(s)-len("_cb-0")])
            else:
                stims.append(s)
        trial = datafile['data']['trial'][:, 0]
        pids = datafile['pids']
        datafile.close()
        mask = np.array([p not in exclude for p in pids])
        pids = [x for x in pids if x not in exclude]
        columns = []
        if itrial:
            columns.append(trial)
        if istim:
            columns.append(stims)
        df = pd.DataFrame(data.T[mask], columns=columns, index=pids)
        dfs.append(df)
        
    df = pd.concat(dfs)
    colnames = []
    if itrial:
        colnames.append("trial")
    if istim:
        colnames.append("stimulus")
    df.columns.names = colnames
    df.index.names = ["pid"]
    return df

def process_model_turk(hstim, nthresh0, nthresh):
    rawtruth0, rawipe0, rawsstim, kappas = load_model("stability")

    sstim = np.array(rawsstim)
    idx = np.nonzero((sstim[:, None] == hstim[None, :]))[0]
    assert idx.size > 0

    nfell = (rawtruth0[idx]['nfellA'] + rawtruth0[idx]['nfellB']) / 10.0
    feedback = (nfell > nthresh0)[:, :, 0].T
    feedback[np.isnan(feedback)] = 0.5

    nfell = (rawipe0[idx]['nfellA'] + rawipe0[idx]['nfellB']) / 10.0
    #ipe_samps = nfell.copy()
    ipe_samps = (nfell > nthresh).astype('f8')
    ipe_samps[np.isnan(nfell)] = 0.5

    return (rawipe0[idx], ipe_samps, 
            rawtruth0[idx][:, :, 0].T, feedback, 
            kappas)

# def load_turk(condition, mode="experiment"):
#     path = "../../turk-experiment/turk_%s_data~%s.npz" % (mode, condition)
#     print "Loading '%s'" % path
#     hdata = np.load(path)
#     rawhuman = hdata['data']['response'][..., None]
#     rawhstim = np.array([x.split("~")[0] for x in hdata['stims']])
#     raworder = hdata['data']['trial'][..., None]
#     hdata.close()
#     return rawhuman, rawhstim, raworder    

def process_turk(human0, hstim0, truth0, ipe0, sstim0):
    
    idx = np.nonzero((sstim0[:, None] == hstim0[None, :]))[0]
    truth = truth0[idx].copy()
    ipe = ipe0[idx].copy()

    # human, stimuli, sort, truth, ipe = order_by_trial(
    #     rawhdata, rawhstim, raworder, rawtruth, rawipe)
    # truth = truth[0]
    # ipe = ipe[0]

    return human, stimuli, sort, truth, ipe

#@memory.cache
def order_by_trial(human, stimuli, order, truth, ipe):

    n_subjs = human.shape[1]
    n_stim = stimuli.size
    n_rep = human.shape[2]

    htrial = human.transpose((1, 0, 2)).reshape((n_subjs, -1))
    horder = order.transpose((1, 0, 2)).reshape((n_subjs, -1))

    shape = np.ones((n_stim, n_rep))
    sidx = list((np.arange(n_stim)[:, None] * shape).ravel())
    
    ho = horder[0]
    assert (ho == horder).all()

    sort = np.argsort(ho)
    sorder = ho[sort]
    sstim = stimuli[..., sidx, :, :][..., sort, :, :]
    struth = truth[..., sidx, :, :][..., sort, :, :]
    sipe = ipe[..., sidx, :, :][..., sort, :, :]

    trial_human = None
    trial_order = sort[None].copy()
    trial_stim = sstim[None].copy()
    trial_truth = struth[None].copy()
    trial_ipe = sipe[None].copy()
        
    for hidx in xrange(n_subjs):
        hd = htrial[hidx]
        shuman = hd[sort]
        if trial_human is None:
            trial_human = np.empty((n_subjs,) + shuman.shape, dtype=shuman.dtype)
        trial_human[hidx] = shuman

    out = (trial_human, trial_stim, trial_order, trial_truth, trial_ipe)
    return out

@memory.cache
def summarize(samps, mass, assigns, mthresh=0.095):
    # calculate individual block displacements
    pos0 = samps[..., 1, :3].transpose((1, 0, 2, 3, 4))
    posT = samps[..., 2, :3].transpose((1, 0, 2, 3, 4))
    posT[np.any(posT[..., 2] < mthresh, axis=-1)] = np.nan
    posdiff = posT - pos0

    m = mass[0,0].transpose((1, 0, 2, 3, 4))
    a = assigns[0,0].transpose((1, 0, 2, 3, 4))

    # calculate center of mass displacement
    towermass = np.sum(m, axis=-2)[..., None, :]
    com0 = np.sum(pos0 * m / towermass, axis=-2)
    comT = np.sum(posT * m / towermass, axis=-2)
    comdiff = comT - com0

    # calculate the number of each type of block that fell
    fellA = np.abs(((a == 0) * posdiff)[..., 2]) > mthresh
    fellB = np.abs(((a == 1) * posdiff)[..., 2]) > mthresh
    assert (~(fellA & fellB)).all()
    nfellA = np.sum(fellA, axis=-1).astype('f8')
    nfellB = np.sum(fellB, axis=-1).astype('f8')
    nfellA[np.isnan(comdiff).any(axis=-1)] = np.nan
    nfellB[np.isnan(comdiff).any(axis=-1)] = np.nan

    return posdiff, comdiff, nfellA, nfellB

@memory.cache
def load_human(predicate):
    if predicate == 'stability':
        exp_ver = 7
    elif predicate == 'direction':
        exp_ver = 8
    else:
        raise ValueError, predicate

    # Human
    rawhuman, rawhstim, rawhmeta, raworder = tat.load_human(
        exp_ver=exp_ver, return_order=True)
    n_subjs, n_reps = rawhuman.shape[1:]
    hids = rawhmeta['dimvals']['id']

    return rawhuman, rawhstim, raworder
    
@memory.cache
def load_model(predicate):
    
    if predicate == 'stability':
        sim_ver = 14 #6
    elif predicate == 'direction':
        sim_ver = 15 #7
    else:
        raise ValueError, predicate

    # Model
    # mname = "ipe_data.npz"
    # if not os.path.exists(mname):
    rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=sim_ver)#-2)
    sigmas = rawsmeta["sigmas"]
    phis = rawsmeta["phis"]
    kappas = rawsmeta["kappas"]

    # Truth samples
    sigma0 = list(sigmas).index(0.0)
    phi0 = list(phis).index(0.0)
    truth = rawmodel[sigma0, phi0][:, :, [0]]
        
    # IPE samples
    sigma1 = list(sigmas).index(0.05)#0.04)
    ipe = rawmodel[sigma1, phi0]

    # Summarize data
    mass, cpoids, assigns, intassigns = tat.stimnames2mass(
        rawsstim, kappas)
    posdiff_true, comdiff_true, nfellA_true, nfellB_true = summarize(
        truth, mass, assigns)
    posdiff_ipe, comdiff_ipe, nfellA_ipe, nfellB_ipe = summarize(
        ipe, mass, assigns)
    
    # Save it
    dtype = np.dtype([
        ('nfellA', 'f8'),
        ('nfellB', 'f8'),
        ('comdiff', [('x', 'f8'), ('y', 'f8'), ('z', 'f8')])])
    
    data_true = np.empty(nfellA_true.shape, dtype=dtype)
    data_true['nfellA'] = nfellA_true
    data_true['nfellB'] = nfellB_true
    data_true['comdiff']['x'] = comdiff_true[..., 0]
    data_true['comdiff']['y'] = comdiff_true[..., 1]
    data_true['comdiff']['z'] = comdiff_true[..., 2]
    
    data_ipe = np.empty(nfellA_ipe.shape, dtype=dtype)
    data_ipe['nfellA'] = nfellA_ipe
    data_ipe['nfellB'] = nfellB_ipe
    data_ipe['comdiff']['x'] = comdiff_ipe[..., 0]
    data_ipe['comdiff']['y'] = comdiff_ipe[..., 1]
    data_ipe['comdiff']['z'] = comdiff_ipe[..., 2]

    # np.savez(mname, truth=data_true, ipe=data_ipe, kappas=kappas, stims=rawsstim)

    # else:
    #     data = np.load(mname)
    #     data_true = data['truth']
    #     data_ipe = data['ipe']
    #     rawsstim = data['stims']
    #     kappas = data['kappas']

    return data_true, data_ipe, rawsstim, kappas
    
def load(predicate):
    rawhuman, rawhstim, raworder = load_human(predicate)
    data_true, data_ipe, rawsstim, kappas = load_model(predicate)
    return rawhuman, rawhstim, raworder, data_true, data_ipe, kappas

@memory.cache
def make_observer_data(nthresh0, nthresh, nsamps, order=True):

    out = load('stability')
    rawhuman, rawhstim, raworder, rawtruth, rawipe, kappas = out

    # # new data
    # hdata = np.load("../../turk-experiment/data.npz")
    # rawhuman = hdata['data']['response'][..., None]
    # rawhstim = np.array([x.split("~")[0] for x in hdata['stims']])
    # raworder = hdata['data']['trial'][..., None]
    # idx = np.nonzero((rawhstim0[:, None] == rawhstim[None, :]))[0]
    # rawtruth = rawtruth0[idx].copy()
    # rawipe = rawipe0[idx].copy()
    
    if order:
        human, stimuli, sort, truth, ipe = order_by_trial(
            rawhuman, rawhstim, raworder, rawtruth, rawipe)
        truth = truth[0]
        ipe = ipe[0]
    else:
        truth = rawtruth.copy()
        ipe = rawipe.copy()

    ipe_samps = np.concatenate([
        ((ipe['nfellA'] + ipe['nfellB']) > nthresh).astype(
            'i8')[..., None],
        ], axis=-1)[..., :nsamps, :]
    feedback = np.concatenate([
        ((truth['nfellA'] + truth['nfellB']) > nthresh0).astype(
            'i8'),
        ], axis=-1)

    return feedback, ipe_samps

######################################################################
# Plotting functions

def make_cmap(name, c1, c2, c3):

    colors = {
        'red'   : (
            (0.0, c1[0], c1[0]),
            (0.50, c2[0], c2[0]),
            (1.0, c3[0], c3[0]),
            ),
        'green' : (
            (0.0, c1[1], c1[1]),
            (0.50, c2[1], c2[1]),
            (1.0, c3[1], c3[1]),
            ),
        'blue'  : (
            (0.0, c1[2], c1[2]),
            (0.50, c2[2], c2[2]),
            (1.0, c3[2], c3[2])
            )
        }
    
    cmap = matplotlib.colors.LinearSegmentedColormap(name, colors, 1024)
    return cmap

def save(path, fignum=None, close=True, width=None, height=None, ext=None):
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

        print "Saving figure to %s...'" % (
            os.path.join(directory, name)),
        plt.savefig(os.path.join(directory, name))
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

def plot_polar():

    plt.figure(1)
    for i in xrange(384):
        plt.clf()
        ax = plt.axes(polar=True)
        ax.pcolormesh(
            t, r,
            #1.2**np.log(z[i, 0]),
            #1.1**np.log(bx[i, 0]),
            bx[i,0],
            cmap='gray',
            vmin=0, vmax=0.03)
        ax.set_rmax(r[-1] + 1)
        ax.set_rmin(r[0] - 1)
        pdb.set_trace()

def bootcorr(arr1, arr2, nboot=1000, nsamp=None, with_replacement=True):
    nsubj1, ntrial = arr1.shape
    nsubj2, ntrial = arr2.shape
    corrs = np.empty(nboot)

    if nsamp is None:
        nsamp = min(nsubj1, nsubj2) / 2

    for i in xrange(corrs.size):
        if with_replacement:
            idx1 = np.random.randint(0, nsubj1, nsamp)
            idx2 = np.random.randint(0, nsubj2, nsamp)
        else:
            idx1 = np.random.permutation(nsubj1)[:nsamp]
            idx2 = np.random.permutation(nsubj2)[:nsamp]

        group1 = arr1[idx1].mean(axis=0)
        group2 = arr2[idx2].mean(axis=0)
        corrs[i] = xcorr(group1, group2)

    return corrs

def bootcorr_wc(arr, nboot=1000, nsamp=None, with_replacement=False):
    nsubj, ntrial = arr.shape
    corrs = np.empty(nboot)

    if nsamp is None:
        nsamp = nsubj / 2

    for i in xrange(corrs.size):
        if with_replacement:
            idx1 = np.random.permutation(nsubj)[:nsamp]
            idx2 = np.random.permutation(nsubj)[:nsamp]
        else:
            idx1, idx2 = np.array_split(np.random.permutation(nsubj)[:nsamp*2], 2)

        group1 = arr[idx1].mean(axis=0)
        group2 = arr[idx2].mean(axis=0)
        corrs[i] = xcorr(group1, group2)

    return corrs

def make_truth_df(rawtruth, rawsstim, kappas, nthresh0):
    nfell = (rawtruth['nfellA'] + rawtruth['nfellB']) / 10.0
    truth = nfell > nthresh0
    truth[np.isnan(nfell)] = 0.5
    df = pd.DataFrame(truth[..., 0].T, index=kappas, columns=rawsstim)
    return df

# def make_ipe_df(rawipe, rawsstim, kappas, nthresh):
#     nfell = (rawipe['nfellA'] + rawipe['nfellB']) / 10.0
#     samps = (nfell > nthresh).astype('f8')
#     # samps = nfell.copy()
#     samps[np.isnan(nfell)] = 0.5
#     alpha = np.sum(samps, axis=-1) + 0.5
#     beta = np.sum(1-samps, axis=-1) + 0.5
#     pfell_mean = alpha / (alpha + beta)
#     pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
#     pfell_std = np.sqrt(pfell_var)
#     pfell_meanstd = np.mean(pfell_std, axis=-1)
#     ipe = np.empty(pfell_mean.shape)
#     for idx in xrange(rawipe.shape[0]):
#         x = kappas
#         lam = pfell_meanstd[idx] * 10
#         kde_smoother = mo.make_kde_smoother(x, lam)
#         ipe[idx] = kde_smoother(pfell_mean[idx])
#     df = pd.DataFrame(ipe.T, index=kappas, columns=rawsstim)
#     return df

def plot_smoothing(samps, stims, istim, kappas):
    reload(mo)
    n_trial, n_kappas, n_samples = samps.shape
    alpha = np.sum(samps, axis=-1) + 0.5
    beta = np.sum(1-samps, axis=-1) + 0.5
    pfell_mean = alpha / (alpha + beta)
    pfell_var = (alpha*beta) / ((alpha+beta)**2 * (alpha+beta+1))
    pfell_std = np.sqrt(pfell_var)
    # pfell_meanstd = np.mean(pfell_std, axis=-1)
    y_mean = mo.IPE(np.ones(n_trial), samps, kappas, smooth=True)
    
    colors = cm.hsv(np.round(np.linspace(0, 220, len(istim))).astype('i8'))
    xticks = np.linspace(-1.3, 1.3, 7)
    xticks10 = 10 ** xticks
    xticks10[xticks < 0] = np.round(xticks10[xticks < 0], decimals=2)
    xticks10[xticks >= 0] = np.round(xticks10[xticks >= 0], decimals=1)
    yticks = np.linspace(0, 1, 3)

    plt.suptitle(
        "Likelihood function for feedback given mass ratio",
        fontsize=16)
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
        # lam = pfell_meanstd[i] * 10
        # kde_smoother = mo.make_kde_smoother(x, lam)
        # y_mean = kde_smoother(pfell_mean[i])
        plt.plot(x, y_mean[i],
                 color=colors[idx],
                 linewidth=3)        
        plt.errorbar(x, pfell_mean[i], pfell_std[i], None,
                     color=colors[idx], fmt='o',
                     markeredgecolor=colors[idx],
                     markersize=5,
                     label=str(stims[i]).split("_")[1])
    # plt.legend(loc=8, prop={'size':12}, numpoints=1,
    #            scatterpoints=1, ncol=3, title="Stimuli")

def plot_belief(fignum, r, c, model_beliefs, kappas, cmap):
    fig = plt.figure(fignum)
    plt.clf()
    ratios = np.round(10 ** np.array(kappas), decimals=1)
    n_kappas = len(kappas)
    n = r*c
    exp = np.e#np.exp(np.log(0.5) / np.log(1./27))    
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
    #kidxs = [0, 3, 6, 10, 13, 16, 20, 23, 26]
    for i, cond in enumerate(sorted(model_beliefs.keys())):
        irow, icol = np.unravel_index(i, (r, c))
        ax = plt.subplot(gs[irow, icol])
        n_trial = model_beliefs[cond].shape[1] - 1
        # kappa = kappas[kidx]
        subjname = cond#"True $r=%s$" % float(ratios[kidx])
        img = plot_theta(
            None, None, ax,
            # np.exp(model_theta[kidx]),
            np.exp(model_beliefs[cond]),
            subjname,
            exp=exp,
            cmap=cmap,
            fontsize=14,
            vmin=0,
            vmax=vmax)
        yticks = np.round(
            np.linspace(0, n_kappas-1, 5)).astype('i8')
        if (i%c) == 0:
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
    #logcticks = np.array([0, 0.001, 0.05, 0.25, 1])
    #cticks = np.exp(np.log(logcticks) * np.log(exp))
    cticks = np.linspace(0, vmax, 5)
    cax = plt.subplot(gs[:, -1])
    cb = fig.colorbar(img, ax=ax, cax=cax, ticks=cticks)
    cb.set_ticklabels(np.round(cticks, decimals=2))#logcticks)
    cax.set_title("$\Pr(r|B_t)$", fontsize=14)

def random_model_lh(conds, n_trial, t0=None, tn=None):
    if t0 is None:
        t0 = 0
    if tn is None:
        tn = n_trial

    n_cond = len(conds)
        
    lh = {}
    for cond in conds:
        obstype, group, fbtype, ratio, cb = parse_condition(cond)
        if group != 'all':
            lh[cond] = np.exp(np.array([np.log(0.5)*(tn-t0)])[None])

    return lh

def block_lh(responses, feedback, ipe_samps, prior, kappas, t0=None,
             tn=None, f_smooth=True, p_ignore=0.0):

    reload(mo)
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
        lh[cond] = np.exp(mo.EvaluateObserver(
            resp, feedback[..., order], ipe_samps[order], 
            kappas, prior=prior, p_ignore=p_ignore, smooth=f_smooth))

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
    for cond in conds:
        if cond not in data:
            continue
        shape = data[cond].shape
        assert len(shape) == 2
        info = []
        for i in xrange(shape[1]):
            shape = data[cond][:, i].shape
            if shape == (1,):
                mean = lower = upper = sumlog = np.log(data[cond][:, i][0])
            else:
                mean = np.mean(data[cond][:, i])
                sem = scipy.stats.sem(data[cond][:, i])
                lower = np.log(mean - sem)
                upper = np.log(mean + sem)
                mean = np.log(mean)
                sumlog = np.sum(np.log(data[cond][:, i]))
            sum = np.sum(data[cond][:, i])
            n = data[cond][:, i].size
            info.append([mean, lower, upper, sumlog, sum, n])
        if len(info) == 1:
            stats[cond] = np.array(info[0])
        else:
            stats[cond] = np.array(info)

    #stats = np.swapaxes(np.array(stats), 0, 1)
    return stats

def plot_explicit_judgments(stats, arr, fbtype=None, ratio=None):
    mean = np.mean(arr, axis=0)
    sem = scipy.stats.sem(arr, axis=0, ddof=1)

         
