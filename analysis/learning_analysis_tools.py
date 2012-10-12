import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib
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
