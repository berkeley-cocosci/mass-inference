"""Helper functions for summarizing model data."""
import numpy as np


def compute_nfell(samps, mass, assigns, mthresh=0.095):
    """Given IPE samples, mass ratios, and mass-to-block assignments,
    compute the number of blocks that fell for each stimulus.

    Parameters
    ----------
    samps: numpy array of IPE samples
    mass: mass ratios for each sample
    assigns: binary assignments of block type
    mthresh (default 0.095): block movement threshold

    Returns
    -------
    nfell: number of blocks that fell

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
