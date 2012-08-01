import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pickle
import scipy.stats
import os
import time

import cogphysics
import cogphysics.lib.circ as circ
import cogphysics.lib.nplib as npl
import cogphysics.lib.rvs as rvs
import cogphysics.lib.stats as stats
import cogphysics.tower.analysis_tools as tat

mthresh = 0.095
zscore = False

def order_by_trial(human, stimuli, order, model):

    n_subjs = human.shape[1]
    n_stim = stimuli.size
    n_rep = human.shape[2]

    htrial = human.transpose((1, 0, 2)).reshape((n_subjs, -1))
    horder = order.transpose((1, 0, 2)).reshape((n_subjs, -1))

    trial_human = []
    trial_order = []
    trial_stim = []
    trial_model = []

    shape = np.ones((n_stim, n_rep))
    sidx = list((np.arange(n_stim)[:, None] * shape).ravel())
    
    for hidx in xrange(n_subjs):
        hd = htrial[hidx].copy()
        ho = horder[hidx].copy()
        
        sort = np.argsort(ho)
        shuman = hd[sort]
        sorder = ho[sort]
        sstim = stimuli[..., sidx, :, :][..., sort, :, :]
        smodel = model[..., sidx, :, :][..., sort, :, :]
        
        trial_human.append(shuman)
        trial_order.append(sort)
        trial_stim.append(sstim)
        trial_model.append(smodel)
        
    trial_human = np.array(trial_human, copy=True)
    trial_order = np.array(trial_order, copy=True)
    trial_stim = np.array(trial_stim, copy=True)
    trial_model = np.array(trial_model, copy=True, dtype=model.dtype)

    out = (trial_human, trial_stim, trial_order, trial_model)
    return out

def random_order(n, shape1, shape2, axis=-1, seed=0):
    tidx = np.arange(n)
    RSO = np.random.RandomState(seed)
    RSO.shuffle(tidx)
    stidx = tidx.reshape(shape1)
    order = stidx * np.ones(shape2)
    return order

def load(predicate):

    if predicate == 'stability':
        exp_ver = 7
        sim_ver = 6
    elif predicate == 'direction':
        exp_ver = 8
        sim_ver = 7
    else:
        raise ValueError, predicate
    
    dtype = np.dtype([
        ('stability_nfell', 'i8'),
        ('stability_pfell', 'i8'),
        ('direction', 'f8'),
        ('radius', 'f8')])
    
    # Human
    rawhuman, rawhstim, rawhmeta, raworder = tat.load_human(
        exp_ver=exp_ver, return_order=True)
    n_subjs, n_reps = rawhuman.shape[1:]
    hids = rawhmeta['dimvals']['id']

    # Model
    if os.path.exists("truth_samples_%s.npy" % predicate):
        truth_samples = np.load("truth_samples_%s.npy" % predicate)
    else:
        rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=sim_ver-2)
        sigmas = rawsmeta["sigmas"]
        phis = rawsmeta["phis"]
        kappas = rawsmeta["kappas"]
        sigma0 = list(sigmas).index(0)
        phi0 = list(phis).index(0)
        mass, cpoids, assigns, intassigns = tat.stimnames2mass(
            rawsstim, kappas)
        rawmodel0 = rawmodel[sigma0, phi0][None, None]
        pfell, nfell, truth_samplesS = tat.process_model_stability(
            rawmodel0, mthresh=mthresh, zscore=zscore, pairs=False)
        dirs, truth_samplesD, dirs_perblock = tat.process_model_direction(
            rawmodel0, mthresh=mthresh, pairs=False, mass=mass)
        radii, truth_samplesR, radii_perblock = tat.process_model_radius(
            rawmodel0, mthresh=mthresh, pairs=False, mass=mass)
        truth_samples = np.empty(truth_samplesS.shape, dtype=dtype)
        truth_samples['stability_nfell'] = truth_samplesS.astype('i8')
        truth_samples['stability_pfell'] = (truth_samplesS > 0).astype('i8')
        truth_samples['direction'] = truth_samplesD
        truth_samples['radius'] = truth_samplesR
        np.save("truth_samples_%s.npy" % predicate, truth_samples)

    rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=sim_ver)
    sigmas = rawsmeta["sigmas"]
    phis = rawsmeta["phis"]
    kappas = rawsmeta["kappas"]
    mass, cpoids, assigns, intassigns = tat.stimnames2mass(
        rawsstim, kappas)
    pfell, nfell, samplesS = tat.process_model_stability(
        rawmodel, mthresh=mthresh, zscore=zscore, pairs=False)
    dirs, samplesD, dirs_perblock = tat.process_model_direction(
        rawmodel, mthresh=mthresh, pairs=False, mass=mass)
    radii, samplesR, radii_perblock = tat.process_model_radius(
        rawmodel, mthresh=mthresh, pairs=False, mass=mass)
    samples = np.empty(samplesS.shape, dtype=dtype)
    samples['stability_nfell'] = samplesS.astype('i8')
    samples['stability_pfell'] = (samplesS > 0).astype('i8')
    samples['direction'] = samplesD
    samples['radius'] = samplesR

    all_model = np.array([truth_samples, samples], dtype=dtype)
    fellsamp = all_model[:, 0, 0].transpose((0, 2, 1, 3))

    assert (rawhstim == rawsstim).all()

    return rawhuman, rawhstim, raworder, fellsamp, (sigmas, phis, kappas)

def plot_theta(nrow, ncol, idx, theta, title, exp=2.718281828459045, ratios=None, cmap='hot'):
    plt.subplot(nrow, ncol, idx)
    plt.cla()
    plt.imshow(
        exp ** np.log(theta.T),
        aspect='auto', interpolation='nearest',
        vmin=0, vmax=1, cmap=cmap)
    plt.xticks([], [])
    plt.ylabel("Mass Ratio")
    if ratios is not None:
        n_kappas = len(ratios)
        plt.yticks(np.arange(n_kappas), ratios)
    plt.title(title)
    plt.draw()
    plt.draw()

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
