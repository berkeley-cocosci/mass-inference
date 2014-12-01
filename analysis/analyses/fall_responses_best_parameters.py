#!/usr/bin/env python

"""
Computes the pearson correlation between model and human judgments on "will it
fall?" for each stimulus for all the different parameter combinations of
sigma/phi. Produces a csv file with the following columns:

    sigma (float)
        perceptual uncertainty
    phi (float)
        force uncertainty
    pearsonr (float)
        pearson correlation between model and people

"""

__depends__ = ["human_fall_responses.csv", "model_fall_responses.h5"]

import os
import util
import pandas as pd
import scipy.stats


def run(dest, results_path):
    # load in the human data
    human = pd\
        .read_csv(os.path.join(results_path, "human_fall_responses.csv"))\
        .groupby(['version', 'block'])\
        .get_group(('GH', 'B'))\
        .pivot('stimulus', 'kappa0', 'median')\
        .stack()
    kappas = human.index.get_level_values('kappa0').unique()

    # load in the table of parameter mappings
    query = util.get_query()
    store = pd.HDFStore(
        os.path.join(results_path, "model_fall_responses.h5"), 'r')
    params = store["/{}/param_ref".format(query)]

    # empty list to put the correlations
    corrs = []

    # load in the model data for each parameter combination, one by one
    for key in store.root._f_getChild("/{}".format(query))._v_children:
        if key == 'param_ref':
            continue

        data = store["{}/{}".format(query, key)]
        sigma = params.ix[key]['sigma']
        phi = params.ix[key]['phi']

        model = data\
            .groupby('block')\
            .get_group('B')\
            .pivot('stimulus', 'kappa0', 'median')[kappas]\
            .stack()

        corrs.append(dict(
            sigma=sigma,
            phi=phi,
            pearsonr=scipy.stats.pearsonr(model, human)[0]))

    results = pd.DataFrame(corrs).set_index(['sigma', 'phi']).sortlevel()
    results.to_csv(dest)
    store.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
