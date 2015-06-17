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


def run(dest, results_path, version):
    # load in the human data
    human = pd\
        .read_csv(os.path.join(results_path, "human_fall_responses.csv"))\
        .groupby(['version', 'block'])\
        .get_group((version, 'B'))\
        .pivot('stimulus', 'kappa0', 'median')\
        .stack()
    kappas = human.index.get_level_values('kappa0').unique()

    # load in the model responses
    store = pd.HDFStore(
        os.path.join(results_path, "model_fall_responses.h5"), 'r')

    # empty list to put the correlations
    corrs = []

    # load in the model data for each parameter combination, one by one
    for query in store.root._v_children:
        print query
        params = store["/{}/param_ref".format(query)]

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
                query=query,
                sigma=sigma,
                phi=phi,
                pearsonr=scipy.stats.pearsonr(model, human)[0]))

    results = pd.DataFrame(corrs).set_index(['query', 'sigma', 'phi']).sortlevel()
    results.to_csv(dest)
    store.close()


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to use responses from')
    args = parser.parse_args()
    run(args.to, args.results_path, args.version)
