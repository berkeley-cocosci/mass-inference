#!/usr/bin/env python

# Built-in
from __future__ import division
import argparse
from itertools import product, izip
import logging
# External
import numpy as np
import pandas as pd
# Local
from snippets import datapackage as dpkg
from mass import DATA_PATH

logger = logging.getLogger("mass.sims")


def block_moved(x, mthresh):
    moved = (x > mthresh).astype(float)
    moved[np.isnan(x)] = np.nan
    return moved


def process_model_nmoved(dp, mthresh=0.0025):
    simulation_metadata = dp.load_resource('simulation_metadata')
    simulations = dp.load_resource('simulations.npy')

    index_names = simulation_metadata['index_names']
    index_levels = simulation_metadata['index_levels']

    # there are three parameters: sigma, phi, kappa
    nparam = 3
    param_names = index_names[:nparam]
    params = [index_levels[p] for p in param_names]
    iparams = [range(len(p)) for p in params]

    names = index_names[nparam:]
    levels = [index_levels[i] for i in names]
    index = pd.MultiIndex.from_tuples(
        [x for x in product(*levels)],
        names=names)

    stats_dict = {}
    for ip, p in izip(product(*iparams), product(*params)):
        logger.info("Processing: %s", zip(param_names, p))
        model = pd.Series(
            simulations[ip].ravel(),
            index=index).unstack('posquat')
        xyz = model[['x', 'y', 'z']]
        if 'timestep_index' in xyz.index.names:
            pos0 = xyz.xs(1, level='timestep_index')
            posT = xyz.xs(-1, level='timestep_index')
        elif 'time' in xyz.index.names:
            pos0 = xyz.xs('0.0', level='time')
            posT = xyz.xs('2.0', level='time')
        movement = ((posT - pos0) ** 2).sum(axis=1).unstack('object')
        moved = movement.apply(block_moved, args=[mthresh])
        n_moved = moved.apply(np.nansum, axis=1)
        amt_moved = movement.apply(np.nansum, axis=1)
        med_moved = movement.apply(np.median, axis=1)
        stats_dict[p] = pd.DataFrame({
            'nfell': n_moved,
            'total movement': amt_moved,
            'median movement': med_moved
        }).stack()

    stats = pd.DataFrame.from_dict(stats_dict).T
    stats.index = pd.MultiIndex.from_tuples(stats.index, names=param_names)
    stats = stats\
        .stack('sample')\
        .stack('stimulus')\
        .reset_index()
    return stats


def process_model_fall(exp, tag, force=False):
    src_dp = DATA_PATH.joinpath("model-raw", "%s_%s.dpkg" % (exp, tag))
    dest_dp = DATA_PATH.joinpath("model", "%s_%s_fall.dpkg" % (exp, tag))

    if not src_dp.exists():
        raise IOError("Package '%s' does not exist" % src_dp.relpath())

    if dest_dp.exists() and not force:
        return

    # load the raw model data
    logger.info("Loading '%s'", src_dp)
    dp = dpkg.DataPackage.load(src_dp)
    dp.load_resources()

    # compute the number of blocks that moved
    resp = process_model_nmoved(dp)

    # the destination datapackage already exists, so just load it and
    # update it
    if dest_dp.exists():
        new_dp = dpkg.DataPackage.load(dest_dp)
        new_dp.bump_minor_version()

        r1 = new_dp.get_resource("model.csv")
        r2 = new_dp.get_resource("metadata")

    # the destination datapackage doesn't exist, so we need to create
    # it from scratch
    else:
        new_dp = dpkg.DataPackage(name=dest_dp.name, licenses=['odc-by'])
        new_dp['version'] = '1.0.0'
        new_dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
        new_dp.add_contributor("Peter W. Battaglia", "pbatt@mit.edu")
        new_dp.add_contributor("Joshua B. Tenenbaum", "jbt@mit.edu")

        r1 = dpkg.Resource(name="model.csv", fmt="csv", pth="./model.csv")
        r2 = dpkg.Resource(name="metadata", fmt="json")
        r2['mediaformat'] = "application/json"

        new_dp.add_resource(r1)
        new_dp.add_resource(r2)

    # update the resource data
    r1.data = resp
    r2.data = dict(source=src_dp, nfell_min=0, nfell_max=10)

    # create destination folders, if they don't exist
    if not dest_dp.dirname().exists():
        dest_dp.dirname().makedirs_p()

    # save
    new_dp.save(dest_dp.dirname())
    logger.info("Saved to '%s'" % dest_dp.relpath())


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
    process_model_fall(args.exp, args.tag, force=args.force)
