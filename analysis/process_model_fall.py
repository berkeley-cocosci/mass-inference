# Built-in
from __future__ import division
from argparse import ArgumentParser
from datetime import datetime
from itertools import product, izip
import hashlib
# External
from path import path
import json
import numpy as np
import pandas as pd
# Local
from analysis_tools import load_datapackage as load

import pdb


def block_moved(x, mthresh):
    moved = (x > mthresh).astype(float)
    moved[np.isnan(x)] = np.nan
    return moved


def process_model_nmoved(dp, mthresh=0.095):
    index_names = dp['simulation_metadata']['index_names']
    index_levels = dp['simulation_metadata']['index_levels']

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

    nmoved_dict = {}
    for ip, p in izip(product(*iparams), product(*params)):
        print zip(param_names, p)
        model = pd.Series(
            dp['simulations.npy'][ip].ravel(),
            index=index).unstack('posquat')
        xyz = model[['x', 'y', 'z']]
        pos0 = xyz.xs(1, level='timestep_index')
        posT = xyz.xs(-1, level='timestep_index')
        movement = ((posT - pos0) ** 2).sum(axis=1).unstack('object')
        moved = movement.apply(block_moved, args=[mthresh])
        nmoved_dict[p] = moved.apply(np.nansum, axis=1)

    nmoved = pd.DataFrame.from_dict(nmoved_dict).T
    nmoved.index = pd.MultiIndex.from_tuples(nmoved.index, names=param_names)
    nmoved = nmoved.stack('sample').stack('stimulus').reset_index('stimulus')
    nmoved.columns = ['stimulus', 'nfell']
    nmoved = nmoved.reset_index()
    return nmoved


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--dest", metavar="dir", type=str,
        help="destination directory for processed data")
    parser.add_argument(
        "data_package", metavar="pkg", type=str, nargs="+",
        help="path to datapackage")

    args = parser.parse_args()
    dp_paths = [path(x).abspath() for x in args.data_package]
    outdir = path(args.dest).abspath()

    if not outdir.exists():
        outdir.makedirs_p()

    for dp_path in dp_paths:
        print "Loading '%s'..." % dp_path
        dp = load(dp_path)
        print "Processing..."
        resp = process_model_nmoved(dp)

        out_name = "%s_fall" % dp_path.splitpath()[-1]
        out_path = outdir.joinpath(out_name)
        print "Saving '%s'..." % out_path
        if not out_path.exists():
            out_path.makedirs_p()

        # save a csv file of the data
        csv_name = "model.csv"
        csv_path = out_path.joinpath(csv_name)
        resp.to_csv(csv_path)
        with open(csv_path, 'r') as fh:
            csv_file = fh.read()
        csv_hash = hashlib.md5(csv_file).hexdigest()
        csv_size = csv_path.getsize()

        # construct data package metadata
        timenow = datetime.now().isoformat(" ")
        json_path = out_path.joinpath("datapackage.json")
        metadata = {
            # required
            'name': out_name,
            'datapackage_version': '1.0-beta.5',
            'licenses': [
                {
                    'id': 'odc-by',
                    'url': 'http://opendefinition.org/licenses/odc-by',
                }
            ],

            # optional
            'title': None,
            'description': None,
            'homepage': None,
            'version': '1.0.0',
            'sources': [],
            'keywords': None,
            'last_modified': timenow,
            'image': None,
            'contributors': [
                {
                    'name': 'Jessica B. Hamrick',
                    'email': 'jhamrick@berkeley.edu',
                },
                {
                    'name': 'Peter W. Battaglia',
                    'email': 'pbatt@mit.edu',
                },
                {
                    'name': 'Joshua B. Tenenbaum',
                    'email': 'jbt@mit.edu',
                },
            ],

            # information about the actual data
            'resources': [
                {
                    'name': csv_name,
                    'format': 'csv',
                    'path': "./%s" % csv_name,
                    'bytes': csv_size,
                    'hash': csv_hash,
                    'modified': timenow,
                },
                {
                    'name': 'metadata',
                    'format': 'json',
                    'data': {
                        'source': dp_path,
                        'nfell_min': 0,
                        'nfell_max': 10,
                    },
                    'mediaformat': 'application/json'
                }
            ],
        }

        with open(json_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
