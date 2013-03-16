#!/usr/bin/python
""" Script that compresses data further by computing specific features
(e.g., stability).

"""

import cPickle as pickle
import os
import numpy as np
import re
import string

from optparse import OptionParser

import util.summarize as summarize

SIMPATH = "../data/sims/compressed"
DATAPATH = "../data/model/"


def load(sim_ver):

    # Get the meta data
    meta_path = os.path.join(SIMPATH, sim_ver + ".meta")
    with open(meta_path) as fid:
        meta = pickle.load(fid)

    # Get the data
    data_path = os.path.join(SIMPATH, sim_ver + ".npy")
    data = np.load(data_path, mmap_mode='c')
    (n_sigmas, n_phis, n_kappas, n_stims, n_samples,
     n_blocks, n_times, n_states) = data.shape

    # Organize the stimuli and data
    stimuli = np.array(meta["stim"])
    assert n_stims == stimuli.size, "stimuli & data's stimuli axis don't match"

    return data, stimuli, meta


def stimnames2mass(stims, kappas):
    """ Parses mass stimuli names and returns the masses and
    assignments etc.

    """

    # if isinstance(stimnames, str):
    #     stimnames = [stimnames]
    stimnames = np.array(stims)
    shape = stimnames.shape

    rx = re.compile("mass-tower_(\d{5})_([0,1]{10})")

    cpoids = np.empty(shape, dtype=np.int)
    assigns = np.empty(shape + (10,), dtype=np.int)
    intassigns = np.empty(shape, dtype=np.int)
    for i, stimname in enumerate(stimnames.flat):
        # j = npl.ind2sub(shape, i, order="C").flat[0]
        j = np.unravel_index(i, shape)
        m = rx.match(stimname)
        if not m:
            print "invalid stimname: %s" % stimname
        cpoids[j] = int(m.group(1))
        b = m.group(2)
        assigns[j] = np.fromiter(b, dtype=np.int)
        intassigns[j] = string.atoi(b, 2)

    assigns = assigns[None, None, None, :, None, :, None]
    kappas = np.reshape(kappas, (1, 1, len(kappas), 1, 1, 1, 1))
    mass = (10. ** kappas * assigns + (1. - assigns))

    return mass, cpoids, assigns, intassigns


def process(sim_ver, summary_type):
    """Load IPE simulation data, summarize it according to the given
    summary_type, and save the results to file.

    """

    if summary_type == "nfell":
         summary_func = summarize.compute_nfell

    # load the raw model data
    rawmodel, stims, meta = load(sim_ver)
    sigmas = meta["sigmas"]
    phis = meta["phis"]
    kappas = meta["kappas"]

    sigma0 = list(sigmas).index(0.0)
    sigma1 = list(sigmas).index(0.05)
    phi0 = list(phis).index(0.0)

    if sim_ver.startswith('mass'):
        truth = rawmodel[sigma0, phi0][:, :, [0]]
        ipe = rawmodel[sigma1, phi0]

        # summarize data
        mass, cpoids, assigns, intassigns = stimnames2mass(
            stims, kappas)
        truth = summary_func(truth, mass, assigns)[:, :, 0].T / 10.
        ipe = summary_func(ipe, mass, assigns) / 10.

    else:
        raise ValueError("No rules specified for processing simulation "
                         "data! Given version was: %s" % sim_ver)

    # save
    if not os.path.exists(DATAPATH):
        os.makedirs(DATAPATH)
    datapath = os.path.join(DATAPATH, "%s:%s.npz" % (sim_ver, summary_type))
    print "Saving model data to '%s'..." % datapath
    np.savez(datapath,
             truth=truth,
             ipe=ipe,
             stims=stims,
             kappas=kappas)


if __name__ == "__main__":

    usage = "usage: %prog [options] sim_ver"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--nfell", dest="nfell",
        action="store_true", default=False,
        help="use original stability (non-mass) towers")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        raise ValueError("no simulation version specified")
    else:
        sim_ver = args[0]

    summary_types = []
    if options.nfell:
        summary_types.append("nfell")

    for st in summary_types:
        process(sim_ver, st)
