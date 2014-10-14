#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np
from snippets import circstats as cs
from scipy.special import binom


def compute_llh_fall_counterfactual(pfall_df, fall_df):
    """Computes the log likelihood that the tower fell (or not), given the
    probability of falling.

    """
    pfall = np.asarray(pfall_df.unstack('kappa'))
    p0 = pfall[:, 0]
    p1 = pfall[:, 1]
    p = p0 * (1 - p1) / ((p0 * (1 - p1)) + ((1 - p0) * p1))
    p = np.hstack([p[:, None], p[:, None]])
    p = np.hstack([p[:, None], 1 - p[:, None]])
    fall = np.asarray(fall_df)[:, None]
    llh = np.log((p * fall) + ((1 - p) * (1 - fall))).reshape((-1, 2))
    llh_df = pd.DataFrame(
        llh, index=pfall_df.index, columns=fall_df.columns)
    return llh_df


def compute_llh_fall(pfall_df, fall_df):
    """Computes the log likelihood that the tower fell (or not), given the
    probability of falling.

    """
    pfall = np.asarray(pfall_df.ix[fall_df.index])[:, None]
    fall = np.asarray(fall_df)
    llh = np.log((pfall * fall) + ((1 - pfall) * (1 - fall)))
    llh_df = pd.DataFrame(
        llh, index=pfall_df.index, columns=fall_df.columns)
    return llh_df


def compute_llh_nfell_binomial(pfall_df, nfell_df):
    """Compute the log likelihood that n blocks fell, given the
    probability of a block falling, under the assumption of a binomial
    distribution.

    """
    pfall = np.asarray(pfall_df.ix[nfell_df.index])[:, None]
    nfell = np.asarray(nfell_df)
    logcoef = np.log(binom(10, nfell))
    p0 = np.log(pfall) * nfell
    p1 = np.log(1 - pfall) * (10 - nfell)
    llh = logcoef + p0 + p1
    llh_df = pd.DataFrame(
        llh, index=pfall_df.index, columns=nfell_df.columns)
    return llh_df


def compute_llh_nfell_multinomial(p_nfell_df, nfell_df):
    """Compute the log likelihood that n blocks fell, given the
    probability for each of 0-10 blocks falling, under the assumption
    of a multinomial distribution.

    """
    p_nfell_df = p_nfell_df.unstack('nfell')
    p_nfell = np.asarray(p_nfell_df.ix[nfell_df.index])[:, None]
    nfell = np.asarray(nfell_df, dtype=int)
    shape = nfell.shape + p_nfell.shape[-1:]
    idx = np.concatenate([np.mgrid[:shape[0], :shape[1]], nfell[None]], axis=0)
    llh = np.log((p_nfell * np.ones(shape))[list(idx)])
    llh_df = pd.DataFrame(
        llh, index=nfell_df.index, columns=nfell_df.columns)
    return llh_df


def compute_llh_dir(vmpar_df, direction_df):
    """Compute the log likelihood of falling in a particular direction,
    given the von mises distribution over directions.

    """
    stims = vmpar_df['mean'].index.get_level_values('stimulus')
    direction = np.asarray(direction_df.ix[stims])
    bc = np.ones(direction.shape)
    mean = np.asarray(vmpar_df['mean'].ix[stims])[:, None] * bc
    var = np.asarray(vmpar_df['var'].ix[stims])[:, None] * bc
    ix = ~np.isnan(direction)
    llh = np.zeros(mean.shape)
    llh[ix] = cs.vmlogpdf(direction[ix], mean[ix], var[ix])
    llh_df = pd.DataFrame(
        llh,
        index=vmpar_df['mean'].ix[stims].index,
        columns=direction_df.columns)
    return llh_df


def normalize(df, axis=1):
    normed = df.copy()
    normed[:] = util.normalize(np.asarray(df), axis=axis)[1]
    return normed


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


def run(results_path, seed):
    np.random.seed(seed)
    data = util.load_all()

    hyps = [-1.0, 1.0]

    ipe_fall = data['ipe']['C']\
        .P_fall_mean_all[hyps]\
        .reorder_levels(['stimulus', 'sigma', 'phi'])\
        .stack()\
        .sortlevel()
    ipe_nfell = data['ipe']['C']\
        .P_nfell_all[hyps]\
        .reorder_levels(['stimulus', 'sigma', 'phi', 'nfell'])\
        .stack()\
        .sortlevel()
    ipe_direction_mean = data['ipe']['C']\
        .P_dir_mean_all[hyps]\
        .reorder_levels(['stimulus', 'sigma', 'phi'])\
        .stack()\
        .sortlevel()
    ipe_direction_var = data['ipe']['C']\
        .P_dir_var_all[hyps]\
        .reorder_levels(['stimulus', 'sigma', 'phi'])\
        .stack()\
        .sortlevel()
    empirical_fall = data['empirical']['C'].P_fall_mean[hyps].stack()
    fb_fall = data['fb']['C'].fall[hyps]
    fb_fall.columns.name = 'kappa0'
    fb_nfell = data['fb']['C'].nfell[hyps]
    fb_nfell.columns.name = 'kappa0'
    fb_direction = data['fb']['C'].direction[hyps]
    fb_direction.columns.name = 'kappa0'

    store = pd.HDFStore(results_path, mode='w')

    # compute empirical likelihoods
    key = 'fall/empirical/params_0'
    print key
    fall = fb_fall
    pfall = empirical_fall
    llh_fall_empirical = pfall\
        .groupby(level='kappa')\
        .apply(compute_llh_fall, fall)\
        .stack()\
        .unstack('kappa')
    llh_fall_empirical = normalize(llh_fall_empirical)
    store.append(key, llh_fall_empirical)

    key = 'fall_cf/empirical/params_0'
    print key
    llh_fall_empirical_cf = compute_llh_fall_counterfactual(pfall, fall)\
        .stack()\
        .unstack('kappa')
    llh_fall_empirical_cf = normalize(llh_fall_empirical_cf)
    store.append(key, llh_fall_empirical_cf)

    # compute ipe likelihoods
    fall = fb_fall
    pfall = ipe_fall
    llh_fall_ipe = pfall\
        .groupby(level=['sigma', 'phi', 'kappa'])\
        .apply(compute_llh_fall, fall)\
        .stack()\
        .unstack('kappa')
    llh_fall_ipe = normalize(llh_fall_ipe)
    save(llh_fall_ipe, 'fall/ipe', store)

    llh_fall_ipe_cf = pfall\
        .groupby(level=['sigma', 'phi'])\
        .apply(compute_llh_fall_counterfactual, fall)\
        .stack()\
        .unstack('kappa')
    llh_fall_ipe_cf = normalize(llh_fall_ipe_cf)
    save(llh_fall_ipe_cf, 'fall_cf/ipe', store)

    fall = fb_nfell
    pfall = ipe_fall
    llh_nfell_ipe = pfall\
        .groupby(level=['sigma', 'phi', 'kappa'])\
        .apply(compute_llh_nfell_binomial, fall)\
        .stack()\
        .unstack('kappa')
    llh_nfell_ipe_binomial = normalize(llh_nfell_ipe)
    save(llh_nfell_ipe_binomial, 'nfell_binomial/ipe', store)

    fall = fb_nfell
    pfall = ipe_nfell
    llh_nfell_ipe = pfall\
        .groupby(level=['sigma', 'phi', 'kappa'])\
        .apply(compute_llh_nfell_multinomial, fall)\
        .stack()\
        .unstack('kappa')
    llh_nfell_ipe_multinomial = normalize(llh_nfell_ipe)
    save(llh_nfell_ipe_multinomial, 'nfell_multinomial/ipe', store)

    direction = fb_direction
    vmpar = pd.DataFrame({
        'mean': ipe_direction_mean,
        'var': ipe_direction_var
    })
    vmpar.columns.name = 'param'
    llh_dir_ipe = vmpar\
        .groupby(level=['sigma', 'phi', 'kappa'])\
        .apply(compute_llh_dir, direction)\
        .stack()\
        .unstack('kappa')\
        .fillna(0)
    llh_dir_ipe[np.isinf(llh_dir_ipe)] = 0.0
    llh_dir_ipe = normalize(llh_dir_ipe)
    save(llh_dir_ipe, 'direction/ipe', store)

    store.close()

if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
