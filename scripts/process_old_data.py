"""Uses cogphysics data handling functions to load model and old human
data and save it into the 'data' submodule for easier and more
consistent access.

"""

import numpy as np
import os
import pickle
import cogphysics.tower.analysis_tools as tat
from cogphysics import RESOURCE_PATH

## Configuration

model_datadir = "../data/model/"
human_datadir = "../data/old-human/"

# dataname = [
#     "stability-original",
#     "stability-sameheight",
#     "mass-prediction-stability",
#     ]

dataname = ["mass-all"]

##########


def _summarize(samps, mass, assigns, mthresh=0.095):
    """Given IPE samples, mass ratios, and mass-to-block assignments,
    summarize the IPE data. Internal analysis tools function.

    Parameters
    ----------
    samps: numpy array of IPE samples
    mass: mass ratios for each sample
    assigns: binary assignments of block type
    mthresh (default 0.095): block movement threshold

    Returns
    -------
    posdiff: difference in position for each block
    comdiff: difference in center of mass for the tower
    nfellA: number of blocks of one type to fall
    nfellB: number of blocks of the other type to fall

    """

    # calculate individual block displacements
    pos0 = samps[..., 1, :3].transpose((1, 0, 2, 3, 4))
    posT = samps[..., 2, :3].transpose((1, 0, 2, 3, 4))
    posT[np.any(posT[..., 2] < mthresh, axis=-1)] = np.nan
    posdiff = posT - pos0

    m = mass[0, 0].transpose((1, 0, 2, 3, 4))
    a = assigns[0, 0].transpose((1, 0, 2, 3, 4))

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
    nfell = nfellA + nfellB

    return nfell


def load_simulations(predicate):
    """Load IPE simulation data.

    Returns
    -------
    data_true: true outcomes (sigma == 0)
    data_ipe: model samples (sigma > 0)
    rawsstim: stimuli
    kappas: log mass ratios

    """

    if predicate == 'stability-original':
        sim_ver = 0
    elif predicate == 'stability-sameheight':
        sim_ver = 1
    elif predicate == 'mass-prediction-stability':
        sim_ver = 14
    elif predicate == 'mass-all':
        sim_ver = 16
    else:
        raise ValueError, predicate

    # load the raw model data
    rawmodel, stims, meta = tat.load_model(sim_ver=sim_ver)
    sigmas = meta["sigmas"]
    phis = meta["phis"]
    kappas = meta["kappas"]

    sigma0 = list(sigmas).index(0.0)
    sigma1 = list(sigmas).index(0.05)
    phi0 = list(phis).index(0.0)

    if predicate.startswith('stability'):
        pfell, nfell, fell_persample = tat.process_model_stability(
            rawmodel, mthresh=0.095, zscore=False)
        nfell_truth = fell_persample[sigma0, phi0][:, :, 0].astype('f8')
        nfell_ipe = fell_persample[sigma1, phi0].swapaxes(0, 1).astype('f8')

        # get the stimuli and tower numbers
        pth = os.path.join(RESOURCE_PATH, 'cpobj_conv_stability.pkl')
        with open(pth, "r") as fh:
            conv = pickle.load(fh)
        stims = np.array([conv[x] for x in stims])

    elif predicate.startswith('mass'):
        truth = rawmodel[sigma0, phi0][:, :, [0]]
        ipe = rawmodel[sigma1, phi0]

        # summarize data
        mass, cpoids, assigns, intassigns = tat.stimnames2mass(
            stims, kappas)
        nfell_truth = _summarize(truth, mass, assigns)[:, :, 0].T
        nfell_ipe = _summarize(ipe, mass, assigns)

    nfell_truth /= 10.
    nfell_ipe /= 10.

    return nfell_truth, nfell_ipe, stims, kappas


def save_model(predicate):
    out = load_simulations(predicate)
    if out is None:
        return
    truth, ipe, stims, kappas = out
    if not os.path.exists(model_datadir):
        os.makedirs(model_datadir)
    datapath = os.path.join(model_datadir, predicate + ".npz")
    print "Saving model data to '%s'..." % datapath
    np.savez(datapath,
             truth=truth,
             ipe=ipe,
             stims=stims,
             kappas=kappas)


def load_human(predicate):
    if predicate == 'stability-original':
        exp_ver = 0
    elif predicate == 'stability-sameheight':
        exp_ver = 2
    elif predicate == 'mass-prediction-stability':
        exp_ver = 7
    elif predicate == 'mass-all':
        return
    else:
        raise ValueError, predicate

    # load the raw model data
    rawhuman, stims, meta = tat.load_human(exp_ver=exp_ver)
    human, human_nonmean = tat.process_human_stability(rawhuman, zscore=False)

    if predicate.startswith('stability'):
        # get the stimuli and tower numbers
        pth = os.path.join(RESOURCE_PATH, 'cpobj_conv_stability.pkl')
        with open(pth, "r") as fh:
            conv = pickle.load(fh)
        stims = np.array([conv[x] for x in stims])

    return human, human_nonmean, stims


def save_human(predicate):
    out = load_human(predicate)
    if out is None:
        return
    human, human_nonmean, stims = out
    if not os.path.exists(human_datadir):
        os.makedirs(human_datadir)
    datapath = os.path.join(human_datadir, predicate + ".npz")
    print "Saving human data to '%s'..." % datapath
    np.savez(datapath,
             rawhuman=human_nonmean,
             human=human,
             stims=stims)

##############

if __name__ == "__main__":
    if not hasattr(dataname, "__iter__"):
        dataname = [dataname]

    for dn in dataname:
        save_model(dn)
        save_human(dn)