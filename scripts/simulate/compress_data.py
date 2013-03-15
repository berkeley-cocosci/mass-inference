#!/usr/bin/python
""" Script that compresses the data by only keeping stride time steps."""

import cPickle as pickle
import os
import numpy as np

from optparse import OptionParser

SCRIPTPATH = "../../data/sims/meta"
SAVEPATH = "../../data/sims/compressed"


def compress(sim_ver, sigmas0=None, phis0=None, kappas0=None, timeidx=None):

    # Load the simulation script
    script_path = os.path.join(SCRIPTPATH, sim_ver + "_script.pkl")
    with open(script_path, "r") as fh:
        script = pickle.load(fh)

    # Paths to the stimuli that were simulated
    stim_paths = script["loadinfo"]["stim_paths"]
    stim_dir = os.path.split(stim_paths[0])[0]
    new_stims = [os.path.split(s)[1] for s in stim_paths]
    # Map the new names back to the old names
    name_table_path = os.path.join(stim_dir, "name_table.pkl")
    if not os.path.exists(name_table_path):
        raise IOError(
            "Could note load the name conversion table: %s" %
            name_table_path)
    # Load the conversion table between old and new names
    with open(name_table_path, "r") as fh:
        name_table = pickle.load(fh)
    name_table_inv = dict([(name_table[x], x) for x in name_table])
    # Convert the new names back to old names
    old_stims = []
    for new_name in new_stims:
        old_name = name_table_inv[new_name]
        old_stims.append(old_name)
        print "%s --> %s" % (new_name, old_name)

    # Parameters
    # sigma
    sigmas = np.round(script["runinfo"]["sigmas"], decimals=3)
    if sigmas0:
        isigmas = np.searchsorted(sigmas, sigmas0)
        sigmas = sigmas[isigmas]
    else:
        isigmas = range(len(sigmas))
    # phi
    phis = np.round(script["runinfo"]["phis"], decimals=3)
    if phis0:
        iphis = np.searchsorted(phis, phis0)
        phis = phis[iphis]
    else:
        iphis = range(len(phis))
    # kappa
    kappas = np.round(script["runinfo"]["kappas"], decimals=3)
    if kappas0:
        ikappas = np.searchsorted(kappas, kappas0)
        kappas = kappas[ikappas]
    else:
        ikappas = range(len(kappas))
    # record time
    if timeidx is None:
        timeidx = [0, 1, -1]

    n_sigmas = len(sigmas)
    n_phis = len(phis)
    n_kappas = len(kappas)
    n_times = len(timeidx)

    n_stims = len(old_stims)
    n_samples = script["runinfo"]["n_samples"]
    n_blocks = 10
    n_states = 7

    # n_sigmas, n_phis, n_kappas, n_stims,
    # n_samples, n_recordtimes, n_blocks, n_states
    axorder = (0, 1, 2, 3, 4, 6, 5, 7)
    timeax = 4

    # Path where the simulation data and meta data were saved
    sim_root = os.path.join(script["sim_dir"], "%%s.npy")
    # Reduce the data down to one file
    data = np.empty((n_sigmas, n_phis, n_kappas, n_stims, n_samples,
                     n_times, n_blocks, n_states))
    for istim, stim_name in enumerate(new_stims):
        sim_path = sim_root % os.path.splitext(stim_name)[0]
        print "Loading %s..." % sim_path
        # keep pre-noise, post-noise/repel, and final time steps
        data[:, :, :, istim] = np.load(sim_path).take(
            timeidx, axis=timeax).take(
                isigmas, axis=0).take(
                    iphis, axis=1).take(
                        ikappas, axis=2)

    # re-order axes to axorder
    data = data.transpose(axorder)
    # copy deterministic values over the nans
    if 0. in sigmas and 0. in phis:
        isigma0 = np.flatnonzero(sigmas == 0.)
        iphi0 = np.flatnonzero(phis == 0.)
        data[isigma0, iphi0] = np.tile(
            data[isigma0, iphi0, :, :, :1],
            (1, 1, n_samples, 1, 1, 1))

    ## Make meta
    meta = {"sigmas": sigmas,
            "phis": phis,
            "kappas": kappas,
            "timeidx": timeidx,
            "stim": old_stims,
            }
    metaname = "%s.meta" % sim_ver
    npyname = "%s.npy" % sim_ver

    with open(os.path.join(SAVEPATH, metaname), "w") as fid:
        pickle.dump(meta, fid)
    print "Saved to %s" % os.path.join(SAVEPATH, npyname)
    np.save(os.path.join(SAVEPATH, npyname), data)

if __name__ == "__main__":
    usage = "usage: %prog [options] target"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--sigmas", dest="sigmas", action="store",
        help="list of sigmas, separated by commas",
        metavar="SIGMA_CSV")
    parser.add_option(
        "--kappas", dest="kappas", action="store",
        help="list of kappas, separated by commas",
        metavar="KAPPA_CSV")
    parser.add_option(
        "--phis", dest="phis", action="store",
        help="list of phis, separated by commas",
        metavar="PHI_CSV")
    parser.add_option(
        "--times", dest="times", action="store",
        help="list of times, separated by commas",
        metavar="PHI_CSV")
    parser.add_option(
        "-n", "--num-samples", dest="samples", action="store",
        type=int, help="number of samples of each simulation",
        metavar="PHI_CSV")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        raise ValueError("no target directory name specified")
    else:
        target = args[0]

    compress(target,
             sigmas0=options.sigmas,
             phis0=options.phis,
             kappas0=options.kappas,
             timeidx=options.times)
