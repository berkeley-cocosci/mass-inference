#!/usr/bin/env python

# Builtin
import pickle
import os
import re
import warnings
from copy import deepcopy
# External
import numpy as np
# Cogphysics
import cogphysics
import cogphysics.lib.hashtools as ht
# Scenesim
from scenesim.objects.sso import SSO
from scenesim.objects.pso import RBSO
# Local
from snippets import datapackage as dpkg
from mass import DATA_PATH

OLDPATH = DATA_PATH.joinpath("old/old-cogphysics-model-raw")
NEWPATH = DATA_PATH.joinpath("model-raw")

convtable_path = os.path.join(
    cogphysics.RESOURCE_PATH,
    "cpobj_conv_stability.pkl")
with open(convtable_path, "r") as fh:
    convtable = pickle.load(fh)
conv_cache = {}


def convert_name(oldname):
    global conv_cache

    if oldname in conv_cache:
        newname = conv_cache[oldname]
    else:
        name1 = ht.forward_find_hashes(oldname)[-1]
        if name1 in convtable:
            name2 = convtable[name1]

            matches = re.match(r"stability([0-9]+)", name2)
            stimnum = matches.group(1)
            bitstr = "0"*10

        else:
            matches = re.match(r"mass-tower_([0-9]{5})_([01]{10})", stim)
            stimnum = matches.group(1)
            bitstr = matches.group(2)

        newname = "tower_%s_%s" % (stimnum, bitstr)
        conv_cache[oldname] = newname

    return newname


dataset_name_conv = {
    's_tower_mass41D-all': 'mass_prediction_direction',
    's_tower_mass41S-all': 'mass_prediction_stability',
    's_tower_mass_learningS-all': 'mass_learning',
    's_tower_original-all': 'stability_original',
    's_tower_originalSH-all': 'stability_sameheight',
    's_tower_unstable-all': 'stability_unstable',
    's_tower_unstableSH-all': 'stability_unstable_sameheight',
    's_tower_mass_all-all': 'mass_all',
}

stimsets = {
    's_tower_mass41D-all': 'mass-prediction-direction',
    's_tower_mass41S-all': 'mass-prediction-stability',
    's_tower_mass_learningS-all': 'mass-prediction-stability',
    's_tower_original-all': 'stability-original',
    's_tower_originalSH-all': 'stability-sameheight',
    's_tower_unstable-all': 'stability-unstable',
    's_tower_unstableSH-all': 'stability-unstable-sameheight',
    's_tower_mass_all-all': 'mass-all',
}

files = sorted(set([x.splitext()[0] for x in OLDPATH.listdir()]))

for dataset in files:
    oldname = dataset.splitpath()[1]
    if oldname.startswith("README"):
        continue
    if oldname not in dataset_name_conv:
        warnings.warn("Do not know how to handle dataset '%s'" % oldname)
        continue
    name = dataset_name_conv[oldname] + ".dpkg"
    stimset_path = os.path.join(CPO_PATH, stimsets[oldname])

    datapackage_path = os.path.join(NEWPATH, name)
    if datapackage_path.exists():
        print "%s already exists" % datapackage_path
        continue

    old_data_path = dataset + ".npy"
    meta_path = dataset + ".meta"
    script_path = dataset + ".script"

    data = np.load(old_data_path, mmap_mode='c')
    with open(meta_path, 'r') as fh:
        meta = pickle.load(fh)
    with open(script_path, 'r') as fh:
        script = pickle.load(fh)['runinfo']

    print "-"*70
    print "%s --> %s" % (oldname, name)
    print data.shape

    dims = (
        'sigma', 'phi', 'kappa', 'stimulus',
        'sample', 'object', 'timestep_index', 'posquat'
    )

    # make sure data dimensions are what we expect
    if data.ndim == len(dims):
        pass
    elif data.ndim == (len(dims) - 1):
        # create dimension for kappa
        data = data[:, :, None]
    else:
        raise ValueError("unexpected number of dimensions: %d" % data.ndim)

    # convert stimuli names to something coherent
    stims = [convert_name(stim) for stim in meta['stim']]

    # figure out object names
    objs = None
    for stim in stims:
        stim_path = os.path.join(stimset_path, stim + ".cpo")
        if not stim_path.exists():
            raise IOError("cannot find stimulus: %s" % stim)

        cpo = SSO.load_tree(stim_path)
        psos = cpo.descendants(type_=RBSO)
        pso_names = [x.getName() for x in psos]

        if objs is None:
            objs = pso_names
        else:
            if objs != pso_names:
                raise AssertionError("inconsistent pso names")

    # construct the new metadata dictionary
    diminfo = {
        'sigma': map(float, meta['sigmas']),
        'phi': map(float, meta['phis']),
        'kappa': map(float, meta.get('kappas', [0.0])),
        'stimulus': stims,
        'sample': range(data.shape[dims.index('sample')]),
        'object': objs,
        'timestep_index': list(meta['timeidx']),
        'posquat': ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3'],
    }

    print "sigma:", diminfo['sigma']
    print "phi:", diminfo['phi']
    print "kappa:", diminfo['kappa']

    # check dimension sizes
    for i, dim in enumerate(dims):
        dimsize = data.shape[i]
        dimvals = diminfo[dim]
        if dimsize != len(dimvals):
            raise AssertionError(
                "%s metadata does not match dimension size" % dim)

    assert diminfo['sigma'] == list(script['sigmas'])
    assert diminfo['phi'] == list(script['phis'])
    assert diminfo['kappa'] == list(script['kappas'])
    assert len(diminfo['sample']) == script['n_samples']

    # extract physics simulation information
    record_steps = ['pre-repel', 0]
    record_steps.extend(list(np.cumsum(script['record_intervals'])))
    timestep_dur = script['stepsize']
    sim_dur = record_steps[-1] * timestep_dur
    force_dur = script['force_dur'] * timestep_dur
    gravity = map(float, script['gravity'])

    # construct dictionary of simulation metadata
    sim_meta = {
        'index_names': dims,
        'index_levels': diminfo,
        'gravity': gravity,
        'simulation_duration': sim_dur,     # in seconds
        'timestep_duration': timestep_dur,  # in seconds
        'force_duration': force_dur,        # in seconds
        'timesteps': record_steps,
    }

    # extract array of force vectors
    force = script['force_xyz']
    force_diminfo = deepcopy(diminfo)
    force_diminfo['vector'] = ['x', 'y', 'z']
    del force_diminfo['stimulus']
    del force_diminfo['object']
    del force_diminfo['timestep_index']
    del force_diminfo['posquat']
    force_dims = [x for x in dims if x in force_diminfo] + ['vector']

    # construct dictionary of force vector metadata
    force_meta = {
        'index_names': force_dims,
        'index_levels': force_diminfo
    }

    # create the datapackage
    dp = dpkg.DataPackage(name=name, licenses=['odc-by'])
    dp['version'] = '1.0.0'
    dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
    dp.add_contributor("Peter W. Battaglia", "pbatt@mit.edu")
    dp.add_contributor("Joshua B. Tenenbaum", "jbt@mit.edu")

    dp.add_resource(dpkg.Resource(
        name="simulations.npy", fmt="npy",
        data=data, pth="./simulations.npy"))

    dp.add_resource(dpkg.Resource(
        name="forces.npy", fmt="npy",
        data=force, pth="./forces.npy"))

    sm = dpkg.Resource(name="simulation_metadata", fmt="json", data=sim_meta)
    sm['mediaformat'] = 'application/json'
    dp.add_resource(sm)

    fm = dpkg.Resource(name="force_metadata", fmt="json", data=force_meta)
    fm['mediaformat'] = 'application/json'
    dp.add_resource(fm)

    dp.save(NEWPATH)
