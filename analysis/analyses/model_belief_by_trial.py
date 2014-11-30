#!/usr/bin/env python

"""
Computes the model belief for each participant for the following models:
    
    * static
    * learning

This depends on RESULTS_PATH/model_likelihood_by_trial.h5, and will produce a
new database similar to that one. For each key in the likelihood database, this
will have keys named <key>/static, <key>/learning, and <key>/static. For each
one of these tables, the columns are:

    version (string)
        the experiment version
    kappa0 (float)
        true log mass ratio
    pid (string)
        unique participant id
    stimulus (string)
        stimulus name
    trial (int)
        trial number
    hypothesis (float)
        hypothesis about the mass ratio
    logp (float)
        log posterior probability of the hypothesis

"""

import os
import sys
import util
import pandas
import numpy

from IPython.parallel import Client, require


def model_belief(args):
    key, old_store_pth, pth = args
    print key

    # make sure the directory where util.py is, is on the path
    if pth not in sys.path:
        sys.path.append(pth)

    # import util, so we can use the normalize functon
    import util

    # load in the model likelihoods
    old_store = pandas.HDFStore(old_store_pth, mode='r')
    data = old_store[key]
    old_store.close()

    # compute the belief
    llh = data\
        .set_index(['pid', 'hypothesis', 'trial'])['llh']\
        .unstack('hypothesis')\
        .sortlevel()

    # compute the belief for each model
    models = {
        'static': llh.copy(),
        'learning': llh.groupby(level='pid').apply(numpy.cumsum),
    }

    for model_name, model in models.items():
        # normalize the probabilities so they sum to one
        model[:] = util.normalize(
            numpy.asarray(model), axis=1)[1]

        # convert to long form
        model = pandas.melt(
            model.reset_index(),
            id_vars=['pid', 'trial'],
            var_name='hypothesis',
            value_name='logp')

        # merge with the existing data
        model = pandas.merge(data, model).drop('llh', axis=1)

        # update the version in the dictionary
        models[model_name] = model

    return key, models


def run(dest, results_path, parallel):
    old_store_pth = os.path.abspath(os.path.join(
        results_path, 'model_likelihood_by_trial.h5'))
    old_store = pandas.HDFStore(old_store_pth, mode='r')
    store = pandas.HDFStore(dest, mode='w')

    # path to the directory with analysis stuff in it
    pth = os.path.abspath(os.path.dirname(__file__))

    if parallel:
        rc = Client()
        lview = rc.load_balanced_view()
        task = require('numpy', 'pandas', 'sys')(model_belief)
    else:
        task = model_belief

    results = []

    for key in old_store.keys():
        if key.split('/')[-1] == 'param_ref':
            store.append(key, old_store[key])
            continue

        args = [key, old_store_pth, pth]
        if parallel:
            result = lview.apply(task, args)
        else:
            result = task(args)
        results.append(result)

    while len(results) > 0:
        result = results.pop(0)
        if parallel:
            key, data = result.get()
            result.display_outputs()
        else:
            key, data = result

        for model in data:
            store.append("{}/{}".format(key, model), data[model])

    store.close()
    old_store.close()

if __name__ == "__main__":
    parser = util.default_argparser(__doc__, results_path=True, parallel=True)
    args = parser.parse_args()
    run(args.dest, args.results_path, args.parallel)
