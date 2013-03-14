#!/usr/bin/env python
""" Makes a simulation script for the tower scenes.

Written by Peter Battaglia (pbatt@mit.edu)
"""

import os
import cPickle as pickle
import numpy as np
import cogphysics.tower.tools as tt

from optparse import OptionParser

STIMPATH = "../../stimuli/obj/new"
LISTPATH = "../../stimuli/lists"
SCRIPTPATH = "../../data/sims/meta"
DATAPATH = "../../data/sims/raw"


def makeScript(stype, target, sigmas, phis, kappas, n_samples):

    RSO = np.random.RandomState(0)

    # Path to the stimulus files
    stim_root = os.path.join(STIMPATH, stype)
    # Path to the simulation's save output directory
    sim_dir = os.path.join(DATAPATH, target)
    # Path to where the script will be saved
    script_path = os.path.join(SCRIPTPATH, target + "_script.pkl")

    # Locations of stimuli
    stim_paths, floor_path = tt.get_scene_paths(stim_root)
    print "\n".join(stim_paths)

    n_sigmas = sigmas.size
    n_kappas = kappas.size

    angs = RSO.rand(n_samples) * 2. * np.pi
    vecs0 = np.vstack((np.cos(angs), np.sin(angs), np.zeros_like(angs)))
    force_xyz = (phis[None, :, None, None, None] *
                 np.ones((n_sigmas, 1, n_kappas, n_samples, 1)) *
                 vecs0.T[None, None, None, ...])

    # Single simulation run parameters
    sim_duration = 2.
    stepsize = 1. / 1000.
    max_steps = int(np.ceil(sim_duration / stepsize))
    strides = 100
    force_dur = 0.200 / stepsize
    gravity = np.array((0., 0., -9.8))

    # Recording info
    record_steps = np.unique(np.hstack(
        (np.arange(0, max_steps, strides, dtype=np.int), max_steps)))
    record_intervals = np.diff(record_steps)

    # Instructions for what simulations.py needs to load (stimuli etc)
    loadinfo = {
        "stim_paths": stim_paths,
        "floor_path": floor_path,
            }

    # Instructions for how to set up each simulation.
    runinfo = {
        "n_samples": n_samples,
        "sigmas": sigmas,
        "phis": phis,
        "kappas": kappas,
        "force_xyz": force_xyz,
        "force_dur": force_dur,
        "gravity": gravity,
        "record_intervals": record_intervals,
        "stepsize": stepsize,
        }

    # Pickle and save
    info = {
        "loadinfo": loadinfo,
        "runinfo": runinfo,
        "sim_dir": sim_dir
        }
    if not os.path.exists(SCRIPTPATH):
        os.makedirs(SCRIPTPATH)
    with open(script_path, "w") as fid:
        pickle.dump(info, fid)

if __name__ == "__main__":
    usage = "usage: %prog options target"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-s", "--stype", dest="stype", action="store",
        help="stimulus type, e.g. mass-learning [required]",
        metavar="STIM_TYPE")
    parser.add_option(
        "--sigmas", dest="sigmas", action="store",
        help="list of sigmas, separated by commas [required]",
        metavar="SIGMA_CSV")
    parser.add_option(
        "--kappas", dest="kappas", action="store",
        help="list of kappas, separated by commas [required]",
        metavar="KAPPA_CSV")
    parser.add_option(
        "--phis", dest="phis", action="store",
        help="list of phis, separated by commas [required]",
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

    if options.stype is None:
        raise ValueError("--stype not given")
    if options.sigmas is None:
        raise ValueError("--sigmas not given")
    if options.kappas is None:
        raise ValueError("--kappas not given")
    if options.phis is None:
        raise ValueError("--phis not given")
    if options.samples is None:
        raise ValueError("--num-samples not given")

    stype = options.stype
    sigmas = np.array([float(x.strip()) for x in options.sigmas.split(",")])
    kappas = np.array([float(x.strip()) for x in options.kappas.split(",")])
    phis = np.array([float(x.strip()) for x in options.phis.split(",")])
    n_samps = options.samples

    makeScript(stype, target, sigmas, phis, kappas, n_samps)


###############################################################
##### These are some old parameter settings, for reference.
###############################################################

# stype  = "tower_mass"
# sigmas    = np.array([0.04])
# phis      = np.array([0.2])
# kappas    = np.array([
#     -0.6, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2])
# n_samples = 48

# #stype  = "tower_mass"
# #target  = "tower_mass"
# stype  = "tower_mass"
# target  = "tower_mass_learning"
# #sigmas    = np.array([0., 0.04, 0.05])
# sigmas = np.array([0, 0.05])
# #phis      = np.array([0., 0.4, 0.8, 1.2, 2.0])
# phis = np.array([0.])
# #kappas    = np.array(
# #    [-0.6, 0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2])
# kappas = np.round(np.arange(-1.3, 1.4, 0.1), decimals=1)
# n_samples = 300
# f_genvis = False

# stype  = "tower_massS"
# target  = "tower_mass_learningS"
# sigmas = np.array([0, 0.05])
# phis = np.array([0.])
# kappas = np.round(np.arange(-1.3, 1.4, 0.1), decimals=1)
# n_samples = 300
# f_genvis = False

# stype  = "tower_massD"
# target  = "tower_mass_learningD"
# sigmas = np.array([0, 0.05])
# phis = np.array([0.])
# kappas = np.round(np.arange(-1.3, 1.4, 0.1), decimals=1)
# n_samples = 300
# f_genvis = False

# stype  = "tower_mass_all"
# target  = "tower_mass_all"
# sigmas = np.array([0, 0.05])
# phis = np.array([0.])
# kappas = np.array([-1.0, -0.3, 0.0, 0.3, 1.0])
# n_samples = 10
# f_genvis = False

# stype = "tower_unstable"
# target = "gv_tower_unstable"
# # stype  = "tower_original"
# # target  = "gv_tower_original"
# sigmas    = np.arange(3)
# # # sigmas = np.arange(11) / 100.
# phis      = np.array([0.2])
# # # phis      = np.array([0., .2, .4, .6, .8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
# kappas    = np.array([0.0])
# n_samples = 48 #16
# # # n_samples = 48
# f_genvis = True # False

# stype = "tower_unstable"
# target = "tower_unstable"
# sigmas    = np.arange(11) / 100.
# phis      = np.array([0., .2, .4, .6, .8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
# kappas    = np.array([0.0])
# n_samples = 48

# stype = "tower_origSH"
# target = "tower_origSH"
# sigmas    = np.arange(11) / 100.
# phis      = np.array([0., .2, .4, .6, .8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
# kappas    = np.array([0.0])
# n_samples = 48

# stype = "tower_unstableSH"
# target = "tower_unstableSH"
# sigmas    = np.arange(11) / 100.
# phis      = np.array([0., .2, .4, .6, .8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
# kappas    = np.array([0.0])
# n_samples = 48
