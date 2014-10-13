#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from path import path


def run(results_path, seed):
    np.random.seed(seed)

    store = pd.HDFStore(path(results_path).dirname().joinpath(
        "model_belief_fit.h5"))

    query = util.get_query()
    sigma, phi = util.get_params()

    empirical = []
    ipe = []

    params = store["{}/ipe/param_ref".format(query)]\
        .reset_index()\
        .set_index(['sigma', 'phi'])['index']\
        .ix[(sigma, phi)]
    ipe_pth = "/{}/ipe/{}".format(query, params)
    empirical_pth = "/{}/empirical/params_0".format(query)

    for key in store.keys():
        if not key.endswith("params"):
            continue
        if key.startswith(ipe_pth):
            print "ipe", key
            ipe.append(store[key])
        elif key.startswith(empirical_pth):
            print "empirical", key
            empirical.append(store[key])

    ipe = pd.concat(ipe).reset_index()
    ipe['likelihood'] = 'ipe'
    empirical = pd.concat(empirical).reset_index()
    empirical['likelihood'] = 'empirical'

    results = pd\
        .concat([empirical, ipe])\
        .set_index([
            'model', 'likelihood', 'version', 'kappa0', 'pid'])\
        .sortlevel()

    results.to_csv(results_path)
    store.close()


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
