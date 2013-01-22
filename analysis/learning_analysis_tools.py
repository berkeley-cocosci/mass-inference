import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
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

from cogphysics.lib.corr import xcorr, partialcorr

from joblib import Memory
memory = Memory(cachedir="cache", mmap_mode='c', verbose=0)

######################################################################
# Data handling

def load_turk_df(condition, mode="experiment", sort_trials=False):
    path = "../../turk-experiment/data/consolidated_data/%s_data~%s.npz" % (mode, condition)
    print "Loading '%s'" % path
    datafile = np.load(path)
    data = datafile['data']['response']
    # stims = []
    # for stim in datafile['stims']:
    #     if stim.startswith("stability"):
    #         stims.append("s" + stim[len("stability"):])
    #     elif stim.startswith("mass-tower"):
    #         stims.append("mt" + stim.split("_")[1])
    #     else:
    #         stims.append(stim)
    stims = datafile['stims']
    trial = datafile['data']['trial'][:, 0]
    pids = datafile['pids']
    datafile.close()
    df = pd.DataFrame(data.T, columns=[trial, stims], index=pids)
    df.columns.names = ["trial", "stimulus"]
    df.index.names = ["pid"]
    if sort_trials:
        df.sort_index(axis=1, inplace=True)
    return df

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

def plot_theta(nrow, ncol, idx, theta, title, exp=2.718281828459045, cmap='hot', fontsize=12):
    try:
        idx.cla()
    except AttributeError:
        ax = plt.subplot(nrow, ncol, idx)
    else:
        ax = idx
    ax.cla()
    img = ax.imshow(
        exp ** np.log(theta.T),
        aspect='auto', interpolation='nearest',
        vmin=0, vmax=1, cmap=cmap)
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
