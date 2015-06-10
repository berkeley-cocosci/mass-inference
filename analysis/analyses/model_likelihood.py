#!/usr/bin/env python

"""
Computes the model likelihood P(F | S, k), where F is feedback, S is the
stimulus, and k is the mass ratio, for several different likelihoods:

    * empirical likelihood
    * empirical counterfactual likelihood
    * ipe likelihood
    * ipe counterfactual likelihood

Additionally, there may be several different versions of the ipe likelihoods,
for each query type that is present in RESULTS_PATH/model_fall_responses.h5. The
empirical likliehoods are computed from RESULTS_PATH/human_fall_responses.csv.

The results are saved into a HDF5 database, with the following structure:

    <likelihood>/params_<n>

where <likelihood> is the name of the likelihood (for example, 'empirical',
'ipe_percent_fell', etc.) and params_<n> (e.g. 'params_0') is the particular
combination of sigma/phi parameters. For the empirical likelihoods, this is just
params_0 (because there are not actually any sigma/phi parameters). For the IPE
likelihoods, there is additionally a <likelihood>/param_ref array that gives a
mapping between the actual parameter values and their identifiers.

For each array stored in the database, it has the following columns:

    stimulus (string)
        the name of the stimulus
    kappa0 (float)
        the true log mass ratio
    hypothesis (float)
        the hypothesis under consideration
    counterfactual (bool)
        whether the counterfactual likelihood was used
    llh (float)
        the log likelihood

"""

__depends__ = ["fb_B", "human_fall_responses.csv", "model_fall_responses.h5"]
__ext__ = '.h5'

import os
import util
import pandas as pd
import numpy as np


def compute_llh(pfall_df, fall_df):
    """Computes the log likelihood that the tower fell (or not), given the
    probability of falling.

    """
    # number of stims x 1 x number of hypotheses
    p = np.asarray(pfall_df)[:, None]
    # number of stims x number of feedback conditions x 1
    F = np.asarray(fall_df)[:, :, None]
    # compute the log likelihood
    llh = np.log((p * F) + ((1 - p) * (1 - F))).reshape((-1, 2))
    # normalize
    llh_norm = util.normalize(llh, axis=1)[1]
    # put it back in a dataframe
    llh_df = pd.DataFrame(
        llh_norm, index=fall_df.stack().index, columns=pfall_df.columns)
    llh_df.columns.name = 'hypothesis'
    llh_df.index.names = ['stimulus', 'kappa0']
    return llh_df.stack().to_frame('llh').reset_index()


def compute_llh_counterfactual(pfall_df, fall_df):
    """Computes the log likelihood that the tower fell (or not), given the
    probability of falling.

    """
    # compute p(F | S, k)(1 - p(F | S, ~k)) / Z
    # Z = p(F | S, k)(1 - p(F | S, ~k)) + (1 - p(F | S, k))p(F | S, ~k)
    p0, p1 = np.asarray(pfall_df).T
    p = p0 * (1 - p1) / ((p0 * (1 - p1)) + ((1 - p0) * p1))

    # compute new dataframe for probabilities
    pfall_df_cf = pd.DataFrame(
        np.hstack([p[:, None], 1 - p[:, None]]),
        index=pfall_df.index, columns=pfall_df.columns)

    # compute log likelihood
    llh_df = compute_llh(pfall_df_cf, fall_df)
    return llh_df


def save(data, pth, store):
    params = ['sigma', 'phi']
    all_params = {}
    for i, (p, df) in enumerate(data.groupby(level=params)):
        key = '{}/params_{}'.format(pth, i)
        print key
        df2 = df.reset_index(params, drop=True)
        store.append(key, df2)
        all_params['params_{}'.format(i)] = p
    all_params = pd.DataFrame(all_params, index=params).T
    store.append('{}/param_ref'.format(pth), all_params)


def run(dest, results_path, data_path, version):
    hyps = [-1.0, 1.0]

    # load empirical probabilities
    human_responses = pd.read_csv(os.path.join(
        results_path, "human_fall_responses.csv"))
    empirical = human_responses\
        .groupby(['version', 'block'])\
        .get_group((version, 'B'))\
        .pivot('stimulus', 'kappa0', 'median')[hyps]

    # load feedback
    fb = (util.load_fb(data_path)['C']['nfell'] > 1).unstack('kappa')[hyps]

    # load ipe probabilites
    old_store = pd.HDFStore(
        os.path.join(results_path, "model_fall_responses.h5"), mode='r')

    store = pd.HDFStore(dest, mode='w')

    # compute empirical likelihood
    key = '/empirical/params_0'
    print key
    llh_empirical = compute_llh(empirical, fb)
    llh_empirical['counterfactual'] = False
    llh_empirical_cf = compute_llh_counterfactual(empirical, fb)
    llh_empirical_cf['counterfactual'] = True
    store.append(key, llh_empirical)
    store.append(key, llh_empirical_cf)

    # compute likelihoods for each query type
    for key in old_store.keys():
        parts = key.split('/')
        parts[1] = 'ipe_{}'.format(parts[1])
        new_key = '/'.join(parts)

        if key.split('/')[-1] == 'param_ref':
            store.append(new_key, old_store[key])
            continue

        ipe = old_store[key]\
            .groupby('block')\
            .get_group('B')\
            .pivot('stimulus', 'kappa0', 'median')[hyps]

        # compute ipe likelihood
        print new_key
        llh_ipe = compute_llh(ipe, fb)
        llh_ipe['counterfactual'] = False
        llh_ipe_cf = compute_llh_counterfactual(ipe, fb)
        llh_ipe_cf['counterfactual'] = True
        store.append(new_key, llh_ipe)
        store.append(new_key, llh_ipe_cf)

    store.close()
    old_store.close()

if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to use responses from')
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.version)
