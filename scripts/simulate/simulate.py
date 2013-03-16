#!/usr/bin/env python
""" Runs and records tower simulations.

Written by Peter Battaglia (pbatt@mit.edu)
"""

import pandac.PandaModules
from libpanda import Vec3
from pandac.PandaModules import NodePath

import cogphysics.lib.nplib as nplib
import cogphysics.scrap.nodetools as nt
import cogphysics.scrap.stimulustools as st
import cogphysics.tower.tools as tt

import cPickle as pickle
import numpy as np
import os
import time

from optparse import OptionParser

SCRIPTPATH = "../../data/sims/meta"


def add_noise(obase, topgnode, rand, sigma=0.04):
    if sigma > 0.:
        # Add the noise
        blocks = [child for child in topgnode.getChildren()]
        n_blocks = len(blocks)
        noises = np.hstack((rand.randn(n_blocks, 2) * sigma,
                            np.zeros((n_blocks, 1))))
        for block, noise in zip(blocks, noises):
            state = nt.get_state(block, ("posg",))
            state["posg"] += noise
            nt.sync_graphics(block, state)
        # Push to physics
        nt.sync_g2p(blocks)
        # Repel
        obase.repel(500)


def set_kappa(topgnode, kappa=0.0):
    blocks = nt.get_descendants(topgnode, depth=[1])
    ratio = 10 ** kappa
    if topgnode.getName().startswith('mass'):
        mass_types = list(topgnode.getName().split("_")[2])
    for bidx, block in enumerate(blocks):
        pnode = block.getPythonTag("pnode")
        mass_type = mass_types[bidx]
        if mass_type == '0':
            pnode.mass = 2. / (1 + ratio)
        elif mass_type == '1':
            pnode.mass = (2. / (1 + ratio)) * ratio
        else:
            print "no mass type"


def is_deterministic(phi, sigma):
    if sigma == 0 and phi == 0:
        return True
    return False


def make_force(bodies, force, dur):
    def forcefunc(i):
        if i <= dur:
            for body in bodies:
                body.setForce(force)
    return forcefunc


def set_pos(gnode, pos):
    for p, block in zip(pos, nt.get_descendants(gnode, depth=[1])):
        nt.sync_graphics(block, {"posg": p})


def put(state, body):
    pos = body.getPosition()
    quat = body.getQuaternion()
    state[:] = (pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3])


def putdata(data, bodies):
    map(put, data, bodies)

########################################################################


def simulate(sim_ver, f_graphics, f_save):

    rand = np.random.RandomState(0)

    # Path to the simulation script
    script_path = os.path.join(SCRIPTPATH, sim_ver + "_script.pkl")

    # Unpickle the simulation script
    print "Loading simulation script..."
    with open(script_path, "r") as fid:
        info = pickle.load(fid)
    # Path where the simulation data and meta data will be saved.
    sim_dir = info["sim_dir"]
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

    # Load and cache stimuli
    stim_paths = info["loadinfo"]["stim_paths"]
    print "Loading and caching %d stimuli..." % len(stim_paths)
    stims, stim_caches = st.load_and_cache(stim_paths)
    n_stims = len(stims)

    # This is just a clean root for all the stuff in the scene, which we
    # can detach and reparentTo as neeeded
    scenenode = NodePath("scene")
    nt.tag_gnode(scenenode)

    # Load floor
    print "Loading floor..."
    floornode = st.load_and_cache([info["loadinfo"]["floor_path"]])[0][0]
    nt.tag_gnode(floornode)
    floornode.reparentTo(scenenode)
    # Make the floor really big
    state = nt.get_state(floornode, ('scalel',))
    state['scalel'][0] = 50
    state['scalel'][1] = 50
    nt.sync_graphics(floornode, state)
    nt.sync_g2p([floornode])

    # Initialize physics
    print "Init physics..."
    obase, lbase = None, None
    topgnodes = [floornode] + stims
    for gnode in topgnodes:
        # It is easiest to initialize over archives
        arch = nt.archive(gnode)
        # Physics
        pnodes = [pnode for pnode in arch["pnodes"] if pnode]
        obase = st.init_physics(pnodes=pnodes, obase=obase)
        # Graphics
        if f_graphics:
            lbase = st.init_graphics(gnodes=arch["gnodes"], lbase=lbase)

    if f_graphics:
        ######################################
        # Graphics stuff -- DEBUGGING
        lbase.cameras.setPos(1.5, -7, 2.5)
        lbase.cameras.lookAt(0, 0, 0.5)
        scenenode.reparentTo(lbase.rootnode)
        ######################################

    # Handle running using info["runinfo"]
    sigmas = info["runinfo"].get("sigmas", np.array([0.]))
    phis = info["runinfo"].get("phis", np.array([0.]))
    kappas = info["runinfo"].get("kappas", np.array([0.]))
    force_xyz = info["runinfo"]["force_xyz"]
    force_dur = info["runinfo"]["force_dur"]
    gravity = info["runinfo"]["gravity"]
    record_intervals = np.hstack((-1, 0, info["runinfo"]["record_intervals"]))
    stepsize = info["runinfo"]["stepsize"]
    n_samples = info["runinfo"].get("n_samples", 1)
    n_sigmas = sigmas.size
    n_phis = phis.size
    n_kappas = kappas.size
    n_records = record_intervals.size

    f_genvis = "genvis" in info["runinfo"]
    if f_genvis:
        res = np.load(info["runinfo"]["genvis"])
        pos = res["pos"]
        sampleidx = res["sampleidx"]
        n_sigmas = sigmas.size
        n_samples = sampleidx.size

    # Parameter combinations
    print "Generating parameter combinations..."
    params = nplib.vgrid(
        np.arange(n_sigmas),
        np.arange(n_phis),
        np.arange(n_kappas),
        np.arange(n_samples), order='C').T
    n_params = params.shape[0]

    obase.stepsize = stepsize
    obase.set_gravity(gravity)

    floornode.getPythonTag("pnode").enable()

    # Store each block's body (for getting state) and block label
    print "Store block bodies..."
    stimbodies, stimblocks = [], []
    for istim, stim in enumerate(stims):
        arch = nt.archive(stim)
        bodies, blocks = [], []
        for pnode in arch["pnodes"]:
            if pnode:
                name = str(pnode)
                bodies.append(pnode.body)
                blocks.append(name[-1])
        stimbodies.append(bodies)
        stimblocks.append(blocks)

    # The simulation loop
    n_states = 7

    # Loop over stims
    metadata = {}
    for istim, stim in enumerate(stims):

        stimname = tt.filebase(stim_paths[istim])

        sbodies = stimbodies[istim]
        n_blocks = len(sbodies)

        # Set up stim
        stim.reparentTo(scenenode)
        stim_cache = stim_caches[istim]
        stimarch = nt.archive(stim)
        map(lambda x: x.enable(), [pn for pn in stimarch["pnodes"] if pn])

        simname = "%s" % (stimname,)
        out_path = os.path.join(sim_dir, simname + ".npy")

        # Make the metadata dict for this set of samples
        metadata[simname] = {
            "stim_path": stim_paths[istim],
            "blocks": stimblocks[istim],
            "script_path": script_path,
            }

        if not f_save or not os.path.isfile(out_path):
            if f_save:
                fid = open(out_path, "w")

            bodies = sbodies

            if f_graphics:
                ######################################
                # Graphics syncing -- DEBUGGING
                pnodes = [pnode for pnode in stimarch["pnodes"] if pnode]
                sync_nodes = [(pnode.parent, pnode.body) for pnode in pnodes]
                sync = lambda gnode, body: nt.sync_graphics(
                    gnode, {"posg": body.getPosition(),
                            "quatg": body.getQuaternion()})
                ######################################

            # Allocate data storage
            alldata = np.zeros((
                n_sigmas,
                n_phis,
                n_kappas,
                n_samples,
                n_records,
                n_blocks,
                n_states))
            alldata.fill(np.nan)

            t0 = time.time()
            for iparam, simparams in enumerate(params):
                # One simulation time
                t1 = time.time()

                # Select parameter settings
                isigma, iphi, ikappa, isample = simparams
                sigma = sigmas[isigma]
                kappa = kappas[ikappa]
                phi = phis[iphi]

                if is_deterministic(sigma, phi) and isample > 0:
                    continue

                # Select this sample's data
                data = alldata[tuple(simparams)]
                # Set force
                if force_xyz.ndim == 4:
                    force = Vec3(*force_xyz[isigma, iphi, isample])
                else:
                    force = Vec3(*force_xyz[tuple(simparams)])
                # Reset state
                st.uncache(stim_cache)
                # Store pre-noise physics states
                putdata(data[0], bodies)
                # Add noise
                if f_graphics:
                    lbase.render_frame()
                    time.sleep(0.2)
                # if genvis, just set the position directly
                if f_genvis:
                    set_pos(stim, pos[istim, sigma, isample])
                    # Push to physics
                    nt.sync_g2p(nt.get_descendants(stim, depth=[1]))
                    obase.repel(500)
                else:
                    add_noise(obase, stim, rand, sigma=sigma)
                    if f_graphics:
                        map(sync, *zip(*sync_nodes))
                if f_graphics:
                    lbase.render_frame()
                    time.sleep(0.2)

                # Store post-noise physics states
                putdata(data[1], bodies)
                # Set kappa values
                set_kappa(stim, kappa=kappa)

                cumsteps = 0
                for irecord in xrange(2, n_records):
                    steps = record_intervals[irecord]
                    # # Apply force
                    dur = min(steps, force_dur - cumsteps)
                    forcefuncs = (make_force(bodies, force, dur),)
                    cumsteps += steps

                    # Step the physics to the next record step
                    obase.step_n(steps, forcefuncs=forcefuncs)

                    if f_graphics:
                        ######################################
                        # Sync to graphics -- DEBUGGING
                        map(sync, *zip(*sync_nodes))
                        lbase.render_frame()
                        ######################################

                    # Get physics states
                    putdata(data[irecord], bodies)

                print (("[%s] [stim %02d/%d] [sim %02d/%d] "
                        "sigma=%.2f, phi=%.2f, "
                        "kappa=% .1f, sample=%02d, t=%.3f, dt=%.3f") % (
                            stimname,
                            istim+1, n_stims,
                            iparam, len(params),
                            sigma, phi, kappa, isample,
                            time.time() - t0, time.time() - t1))

            if f_save:
                # Write out the data
                np.save(fid, alldata)
                fid.close()

            td = time.time() - t0
            print
            print "Stim %i/%i" % (istim+1, n_stims,)
            print "Time per step %.7f" % (td / (n_records * n_params))
            print "Time for 2 sim secs %.4f" % (td / n_params)
            print "Time for 1 stim %.1f" % td

        # Make and save metadata
        meta_path = os.path.join(sim_dir, "meta.pkl")
        with open(meta_path, "w") as fid:
            pickle.dump(metadata, fid)

        # Break down the stims
        map(lambda x: x.disable(), [pn for pn in stimarch["pnodes"] if pn])
        stim.detachNode()

if __name__ == "__main__":
    usage = "usage: %prog options target"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-g", "--graphics", dest="f_graphics",
        action="store_true", default=False,
        help="display graphical rendering of simulations")
    parser.add_option(
        "--no-save", dest="f_save",
        action="store_false", default=True,
        help="don't save simulation data",
        metavar="SIGMA_CSV")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        raise ValueError("no target directory name specified")
    else:
        target = args[0]

    simulate(target, options.f_graphics, options.f_save)
