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

def order_by_trial(human, stimuli, order, truth, ipe):

    n_subjs = human.shape[1]
    n_stim = stimuli.size
    n_rep = human.shape[2]

    htrial = human.transpose((1, 0, 2)).reshape((n_subjs, -1))
    horder = order.transpose((1, 0, 2)).reshape((n_subjs, -1))

    trial_human = []
    trial_order = []
    trial_stim = []
    trial_truth = []
    trial_ipe = []

    shape = np.ones((n_stim, n_rep))
    sidx = list((np.arange(n_stim)[:, None] * shape).ravel())
    
    for hidx in xrange(n_subjs):
        hd = htrial[hidx].copy()
        ho = horder[hidx].copy()
        
        sort = np.argsort(ho)
        shuman = hd[sort]
        sorder = ho[sort]
        sstim = stimuli[..., sidx, :, :][..., sort, :, :]
        struth = truth[..., sidx, :, :][..., sort, :, :]
        sipe = ipe[..., sidx, :, :][..., sort, :, :]
        
        trial_human.append(shuman)
        trial_order.append(sort)
        trial_stim.append(sstim)
        trial_truth.append(struth)
        trial_ipe.append(sipe)
        
    trial_human = np.array(trial_human, copy=True)
    trial_order = np.array(trial_order, copy=True)
    trial_stim = np.array(trial_stim, copy=True)
    trial_truth = np.array(trial_truth, copy=True, dtype=truth.dtype)
    trial_ipe = np.array(trial_ipe, copy=True, dtype=ipe.dtype)

    out = (trial_human, trial_stim, trial_order, trial_truth, trial_ipe)
    return out

def random_order(n, shape1, shape2, axis=-1, seed=0):
    tidx = np.arange(n)
    RSO = np.random.RandomState(seed)
    RSO.shuffle(tidx)
    stidx = tidx.reshape(shape1)
    order = stidx * np.ones(shape2)
    return order

# def load(predicate):

#     if predicate == 'stability':
#         exp_ver = 7
#         sim_ver = 6
#     elif predicate == 'direction':
#         exp_ver = 8
#         sim_ver = 7
#     else:
#         raise ValueError, predicate
    
#     dtype = np.dtype([
#         ('stability_nfell', 'i8'),
#         ('stability_pfell', 'i8'),
#         ('direction', 'f8'),
#         ('radius', 'f8'),
#         ('x', 'f8'),
#         ('y', 'f8')])
    
#     # Human
#     rawhuman, rawhstim, rawhmeta, raworder = tat.load_human(
#         exp_ver=exp_ver, return_order=True)
#     n_subjs, n_reps = rawhuman.shape[1:]
#     hids = rawhmeta['dimvals']['id']

#     # Model
#     if os.path.exists("truth_samples_%s.npy" % predicate):
#         truth_samples = np.load("truth_samples_%s.npy" % predicate)
#     else:
#         rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=sim_ver-2)
#         sigmas = rawsmeta["sigmas"]
#         phis = rawsmeta["phis"]
#         kappas = rawsmeta["kappas"]
#         sigma0 = list(sigmas).index(0)
#         phi0 = list(phis).index(0)
#         mass, cpoids, assigns, intassigns = tat.stimnames2mass(
#             rawsstim, kappas)
#         rawmodel0 = rawmodel[sigma0, phi0][None, None]
#         pfell, nfell, truth_samplesS = tat.process_model_stability(
#             rawmodel0, mthresh=mthresh, zscore=zscore, pairs=False)
#         dirs, truth_samplesD, dirs_perblock = tat.process_model_direction(
#             rawmodel0, mthresh=mthresh, pairs=False, mass=mass)
#         radii, truth_samplesR, radii_perblock = tat.process_model_radius(
#             rawmodel0, mthresh=mthresh, pairs=False, mass=mass)
#         truth_samplesX = np.cos(truth_samplesD) * truth_samplesR
#         truth_samplesY = np.sin(truth_samplesD) * truth_samplesR
#         truth_samples = np.empty(truth_samplesS.shape, dtype=dtype)
#         truth_samples['stability_nfell'] = truth_samplesS.astype('i8')
#         truth_samples['stability_pfell'] = (truth_samplesS > 0).astype('i8')
#         truth_samples['direction'] = truth_samplesD
#         truth_samples['radius'] = truth_samplesR
#         truth_samples['x'] = truth_samplesX
#         truth_samples['y'] = truth_samplesY
#         np.save("truth_samples_%s.npy" % predicate, truth_samples)

#     if os.path.exists("model_samples_%s.npz" % predicate):
#         data = np.load("model_samples_%s.npz" % predicate)
#         samples = data['samples']
#         sigmas = data['sigmas']
#         phis = data['phis']
#         kappas = data['kappas']
#         rawsstim = data['rawsstim']
        
#     else:
#         rawmodel, rawsstim, rawsmeta = tat.load_model(sim_ver=sim_ver)
#         sigmas = rawsmeta["sigmas"]
#         phis = rawsmeta["phis"]
#         kappas = rawsmeta["kappas"]
#         mass, cpoids, assigns, intassigns = tat.stimnames2mass(
#             rawsstim, kappas)
#         pfell, nfell, samplesS = tat.process_model_stability(
#             rawmodel, mthresh=mthresh, zscore=zscore, pairs=False)
#         dirs, samplesD, dirs_perblock = tat.process_model_direction(
#             rawmodel, mthresh=mthresh, pairs=False, mass=mass)
#         radii, samplesR, radii_perblock = tat.process_model_radius(
#             rawmodel, mthresh=mthresh, pairs=False, mass=mass)
#         samplesX = np.cos(samplesD) * samplesR
#         samplesY = np.sin(samplesD) * samplesR
#         samples = np.empty(samplesS.shape, dtype=dtype)
#         samples['stability_nfell'] = samplesS.astype('i8')
#         samples['stability_pfell'] = (samplesS > 0).astype('i8')
#         samples['direction'] = samplesD
#         samples['radius'] = samplesR
#         samples['x'] = samplesX
#         samples['y'] = samplesY

#         np.savez("model_samples_%s.npz" % predicate,
#                  samples=samples,
#                  sigmas=sigmas,
#                  phis=phis,
#                  kappas=kappas,
#                  rawsstim=rawsstim)

#     assert (rawhstim == rawsstim).all()

#     all_model = np.array([truth_samples, samples], dtype=dtype)
#     fellsamp = all_model[:, 0, 0].transpose((0, 2, 1, 3))

#     return rawhuman, rawhstim, raworder, fellsamp, (sigmas, phis, kappas)

def summarize(samps, mass, assigns):
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

def load(predicate):

    if predicate == 'stability':
        exp_ver = 7
        sim_ver = 14 #6
    elif predicate == 'direction':
        exp_ver = 8
        sim_ver = 15 #7
    else:
        raise ValueError, predicate

    # Human
    rawhuman, rawhstim, rawhmeta, raworder = tat.load_human(
        exp_ver=exp_ver, return_order=True)
    n_subjs, n_reps = rawhuman.shape[1:]
    hids = rawhmeta['dimvals']['id']

    # Model
    mname = "ipe_data.npz"
    if not os.path.exists(mname):
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

        np.savez(mname, truth=data_true, ipe=data_ipe, kappas=kappas)

    else:
        data = np.load(mname)
        data_true = data['truth']
        data_ipe = data['ipe']
        kappas = data['kappas']

    return rawhuman, rawhstim, raworder, data_true, data_ipe, kappas

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
        plt.yticks(np.arange(0, n_kappas, 2), ratios[::2], fontsize=8)
    plt.title(title)

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

# def faster_inverse(M):
#     # sanity check the shape
#     oldshape = M.shape
#     assert oldshape[-2] == oldshape[-1]
#     # find out dimensions and reshape array
#     nm = np.prod(oldshape[:-2])
#     n = oldshape[-1]
#     A = M.reshape((nm, n, n))
#     # determin pivots
#     pivots = zeros(n, np.intc)
#     # allocate for inverse
#     AI = np.zeros(A.shape) + np.identity(n)
#     # find inverse of each matrix
#     for i in xrange(nm):
#         results = lapack_lite.dgesv(
#             n, n, A[i], n, np.copy(pivots), AI[i], n, 0)
#         if results['info'] > 0:
#             print 'Warning: %d, Singular matrix' % i
#             AI[i] = np.nan
#     # reshape array to original shape
#     MI = AI.reshape(oldshape)
#     return MI
