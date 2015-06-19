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

__depends__ = ["fb_B", "human_fall_responses_raw.csv", "model_fall_responses_raw.h5"]
__random__ = True

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
    llh_df.index.names = ['sample', 'stimulus', 'kappa0']
    return llh_df.stack().to_frame('llh')


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


def bootsamps(df, n=10000):
    arr = np.asarray(df)
    ix = np.random.randint(0, df.size, (n, df.size))
    if ((arr == 0) | (arr == 1)).all():
        alpha = arr[ix].sum(axis=1) + 0.5
        beta = (1 - arr[ix]).sum(axis=1) + 0.5
        means = alpha / (alpha + beta)
    else:
        means = arr[ix].mean(axis=1)
    s = pd.Series(means, name=df.name)
    s.index.name = 'sample'
    return s


def stats(df):
    return pd.Series(
        np.percentile(df, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'],
        name=df.name)


def bootstrap_llh(llh_func, p, fb, n=10000):
    samps = p\
        .groupby(level=['stimulus', 'kappa'])\
        .apply(bootsamps, n=n)\
        .to_frame('p')\
        .reset_index()

    data = pd\
        .merge(samps, fb)\
        .reset_index()\
        .set_index(['sample', 'stimulus', 'kappa'])\
        .sortlevel()

    pfall_df = data['p'].unstack('kappa')
    fall_df = data['fb'].unstack('kappa')
    llh_samps = llh_func(pfall_df, fall_df)['llh']

    llh = llh_samps\
        .groupby(level=['stimulus', 'kappa0', 'hypothesis'])\
        .apply(lambda x: np.log(stats(np.exp(x))))\
        .unstack()\
        .reset_index()

    return llh


def run(dest, results_path, data_path, version, seed):
    np.random.seed(seed)
    hyps = [-1.0, 1.0]

    # load empirical probabilities
    human_responses = pd.read_csv(os.path.join(
        results_path, "human_fall_responses_raw.csv"))
    empirical = human_responses\
        .groupby(['version', 'block'])\
        .get_group((version, 'B'))\
        .rename(columns={'kappa0': 'kappa'})\
        .set_index(['stimulus', 'kappa', 'pid'])['fall? response']\
        .unstack('kappa')[hyps]\
        .stack()

    # load feedback
    fb = (util.load_fb(data_path)['C']['nfell'] > 1)\
        .unstack('kappa')[hyps]\
        .stack()\
        .to_frame('fb')\
        .reset_index()

    # load ipe probabilites
    old_store = pd.HDFStore(
        os.path.join(results_path, "model_fall_responses_raw.h5"), mode='r')

    # get the parameters we want
    sigma, phi = util.get_params()

    # dataframe to store all the results
    all_llh = pd.DataFrame([])

    # compute empirical likelihood
    print('empirical')
    llh_empirical = bootstrap_llh(compute_llh, empirical, fb)
    llh_empirical['counterfactual'] = False
    llh_empirical['likelihood'] = 'empirical'
    all_llh = all_llh.append(llh_empirical)

    print('empirical cf')
    llh_empirical_cf = bootstrap_llh(compute_llh_counterfactual, empirical, fb)
    llh_empirical_cf['counterfactual'] = True
    llh_empirical_cf['likelihood'] = 'empirical'
    all_llh = all_llh.append(llh_empirical_cf)

    # compute likelihoods for each query type
    for query in old_store.root._v_children:

        # look up the name of the key for the parameters that we want (will be
        # something like params_0)
        param_ref_key = "/{}/param_ref".format(query)
        params = old_store[param_ref_key]\
            .reset_index()\
            .set_index(['sigma', 'phi'])['index']\
            .ix[(sigma, phi)]

        # get the data
        key = "/{}/{}".format(query, params)
        ipe = old_store[key]\
            .groupby('block')\
            .get_group('B')\
            .rename(columns={'kappa0': 'kappa'})\
            .set_index(['stimulus', 'kappa', 'sample'])['response']\
            .unstack('kappa')[hyps]\
            .stack()

        # compute ipe likelihood
        print(query)
        llh_ipe = bootstrap_llh(compute_llh, ipe, fb)
        llh_ipe['counterfactual'] = False
        llh_ipe['likelihood'] = 'ipe_' + query
        all_llh = all_llh.append(llh_ipe)

        print(query + ' cf')
        llh_ipe_cf = bootstrap_llh(compute_llh_counterfactual, ipe, fb)
        llh_ipe_cf['counterfactual'] = True
        llh_ipe_cf['likelihood'] = 'ipe_' + query
        all_llh = all_llh.append(llh_ipe_cf)

    old_store.close()

    results = all_llh\
        .set_index(['likelihood', 'counterfactual', 'stimulus', 'kappa0', 'hypothesis'])\
        .sortlevel()

    assert not np.isnan(results['median']).any()
    assert not np.isinf(results['median']).any()

    results.to_csv(dest)


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    parser.add_argument(
        '--version',
        default=config['analysis']['human_fall_version'],
        help='which version of the experiment to use responses from')
    args = parser.parse_args()
    run(args.to, args.results_path, args.data_path, args.version, args.seed)
