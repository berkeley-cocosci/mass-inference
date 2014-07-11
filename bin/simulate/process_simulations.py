#!/usr/bin/env python

from path import path
from snippets import datapackage as dpkg
import argparse
import numpy as np
import re

from mass.sims.tasks import Tasks
from mass.sims.utils import get_params
from mass import DATA_PATH


def extract_key(taskname):
    return re.match(r"(\w+)_([a-zA-Z0-9\-]+)_(\d{2})", taskname).groups()


def load(exp, tag):
    params = get_params(exp, tag)
    tasks = Tasks.load(params['tasks_path'])

    # first load in all the data
    data_by_key = {}
    conditions = {}
    for taskname in sorted(tasks.keys()):
        task = tasks[taskname]
        data = np.load(task['data_path'])
        stim, _tag, chunk = extract_key(taskname)
        assert _tag == tag
        key = stim
        if key not in data_by_key:
            data_by_key[key] = []
            conditions[key] = []
        data_by_key[key].append(data)
        # the conditions are the actual indexes and values of all the
        # simulations that were run in this chunk. we want to read
        # those in, too, so we can make sure we concatenate the data
        # in the correct order
        conditions[key].append(np.array(task['conditions']))

    index_names = params['index_names']
    index_levels = params['index_levels']

    shape = tuple([
        len(index_levels[x]) if x != 'stimulus' else 1
        for x in index_names])

    stim_axis = index_names.index('stimulus')
    time_axis = index_names.index('timestep')
    time_idx = [0, 1, -1]

    for key in data_by_key:
        # get the simulation indices, and sort by them, so we can
        # correctly order the data
        all_conditions = np.vstack(conditions[key])
        keys = all_conditions[:, :, 0][:, ::-1].T
        order = np.lexsort(keys)
        # concatenate the data, reorder it, and reshape it to have the
        # correct shape, and index into the time axis
        all_data = np.vstack(data_by_key[key])[order].reshape(shape)
        data_by_key[key] = all_data.take(time_idx, axis=time_axis)

    step_size = params['simulation']['step_size']
    times = index_levels['timestep'][:1] + [
        str(int(x) * step_size)
        for x in np.array(index_levels['timestep'])[1:]]
    index_levels['time'] = [times[i] for i in time_idx]
    del index_levels['timestep']
    index_names[index_names.index('timestep')] = 'time'
    index_levels['stimulus'] = [
        str(path(x).namebase) for x in index_levels['stimulus']]

    data = np.concatenate(
        [data_by_key[key] for key in sorted(data_by_key.keys())],
        axis=stim_axis)

    return params, data


def process(exp, tag, overwrite=False):
    name = "%s_%s.dpkg" % (exp, tag)
    dp_path = DATA_PATH.joinpath("model-raw", name)

    if dp_path.exists() and not overwrite:
        return

    params, data = load(exp, tag)

    forces = params['forces']
    noises = params['noises']

    sim_meta = {
        'simulations': params['simulation'],
        'physics': params['physics'],
        'index_names': params['index_names'],
        'index_levels': params['index_levels'],
    }

    force_meta = {'index_names': ['phi', 'stimulus', 'sample']}
    force_meta['index_levels'] = {
        x: params['index_levels'][x] for x in force_meta['index_names']
    }

    noise_meta = {'index_names': [
        'sigma', 'stimulus', 'sample', 'object', 'position'
    ]}
    noise_meta['index_levels'] = {
        x: params['index_levels'][x] for x in noise_meta['index_names'][:-1]
    }
    noise_meta['index_levels']['position'] = ['x', 'y', 'z']

    # load the existing datapackage and bump the version
    if dp_path.exists():
        dp = dpkg.DataPackage.load(dp_path)
        dp.bump_minor_version()
        dp.clear_resources()

    # create the datapackage
    else:
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
        data=forces, pth="./forces.npy"))

    dp.add_resource(dpkg.Resource(
        name="noises.npy", fmt="npy",
        data=noises, pth="./noises.npy"))

    sm = dpkg.Resource(name="simulation_metadata", fmt="json", data=sim_meta)
    sm['mediaformat'] = 'application/json'
    dp.add_resource(sm)

    fm = dpkg.Resource(name="force_metadata", fmt="json", data=force_meta)
    fm['mediaformat'] = 'application/json'
    dp.add_resource(fm)

    fm = dpkg.Resource(name="noise_metadata", fmt="json", data=noise_meta)
    fm['mediaformat'] = 'application/json'
    dp.add_resource(fm)

    dp.save(dp_path.dirname())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-t", "--tag",
        required=True,
        help="Simulation tag. A short label for this simulation config.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force datapackages to be generated.")

    args = parser.parse_args()
    process(args.exp, args.tag, overwrite=args.force)
