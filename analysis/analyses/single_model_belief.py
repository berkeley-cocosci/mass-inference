#!/usr/bin/env python

"""
Pulls out the model belief for the specified IPE likelihood and empirical
likelihood from the database of all model beliefs. Depends on the database
RESULTS_PATH/model_belief_by_trial_fit.h5, and produces a csv file with the
following columns:

    likelihood (string)
        the likelihood name (e.g., ipe, empirical)
    counterfactual (bool)
        whether the counterfactual likelihood was used
    model (string)
        the model name (e.g. static, learning)
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
    p (float)
        fitted posterior probability of the hypothesis that r=10
    p correct (float)
        fitted posterior probability of the correct hypothesis
    p raw (float)
        raw posterior probability of the hypothesis that r=10
    p correct raw (float)
        raw posterior probability of the correct hypothesis
    B (float)
        fitted parameter for the logistic regression

"""

__depends__ = ["model_belief_by_trial_fit.h5"]

import os
import util
import pandas as pd


def load_model(store, pth, name):
    group = store.root._f_getChild(pth)
    all_data = pd.DataFrame([])
    for model in group._v_children:
        data = store["{}/{}".format(pth, model)]
        data['model'] = model
        data['likelihood'] = name
        all_data = all_data.append(data)
    return all_data

def run(dest, results_path):
    # open up the database
    store = pd.HDFStore(os.path.abspath(os.path.join(
        results_path, 'model_belief_by_trial_fit.h5')))

    query = util.get_query()
    sigma, phi = util.get_params()

    # look up the name of the key for the parameters that we want (will be
    # something like params_0)
    params = store["/ipe_{}/param_ref".format(query)]\
        .reset_index()\
        .set_index(['sigma', 'phi'])['index']\
        .ix[(sigma, phi)]

    # load in the data
    data = pd.concat([
        load_model(store, "/ipe_{}/{}".format(query, params), "ipe"),
        load_model(store, "/empirical/params_0", "empirical"),
    ]).set_index(['likelihood', 'counterfactual', 'model'])

    store.close()
    data.to_csv(dest)


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
