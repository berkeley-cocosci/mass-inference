import pandas as pd
import numpy as np
import scipy

def bootstrap_mean(x, nsamples=10000):
    arr = np.asarray(x)
    n, = arr.shape
    boot_idx = np.random.randint(0, n, n * nsamples)
    boot_arr = arr[boot_idx].reshape((n, nsamples))
    boot_mean = boot_arr.mean(axis=0)
    stats = pd.Series(
        np.percentile(boot_mean, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'],
        name=x.name)
    return stats

from mass.model import IPE, EmpiricalIPE, NoFeedback, Feedback
from snippets import datapackage as dpkg

def load_participants(version, data_path):
    exp_dp = dpkg.DataPackage.load(data_path.joinpath(
        "human/mass_inference-%s.dpkg" % version))
    participants = exp_dp.load_resource("participants.csv")
    participants = participants.reset_index()
    return participants


def load_human(version, data_path):
    exp_dp = dpkg.DataPackage.load(data_path.joinpath(
        "human/mass_inference-%s.dpkg" % version))
    exp = exp_dp.load_resource("experiment.csv")

    # convert timestamps into datetime objects
    exp['timestamp'] = pd.to_datetime(exp['timestamp'])

    # split into the three different blocks
    expA = exp.groupby('mode').get_group('experimentA')
    expB = exp.groupby('mode').get_group('experimentB')
    expC = exp.groupby('mode').get_group('experimentC')

    exp_data = {
        'A': expA,
        'B': expB,
        'C': expC,
        'all': exp
    }

    return exp_data


def load_model(version, data_path, sigma=0.04, phi=0.2):
    ipe = {
        'A': IPE("mass_inference-%s-a_ipe_fall" % version,
                 sigma=sigma, phi=phi),
        'B': IPE("mass_inference-%s-b_ipe_fall" % version,
                 sigma=sigma, phi=phi),
        'C': IPE("mass_inference-%s-b_ipe_fall" % version,
                 sigma=sigma, phi=phi)
    }

    fb = {
        'A': NoFeedback("mass_inference-%s-a_truth_fall" % version),
        'B': NoFeedback("mass_inference-%s-b_truth_fall" % version),
        'C': Feedback("mass_inference-%s-b_truth_fall" % version),
    }

    return ipe, fb


def load_all(model_version=None, human_version=None, data_path=None,
             human=None, ipe=None, fb=None, sigma=0.04, phi=0.2):
    if ipe is None or fb is None:
        ipe, fb = load_model(model_version, data_path, sigma=sigma, phi=phi)

    if human is None:
        human = load_human(human_version, data_path)

    data = {
        'human': human,
        'ipe': ipe,
        'fb': fb,
        'empirical': {'C': EmpiricalIPE(human['B'])}
    }

    return data

def beta(x, n=1, percentiles=None):
    arr = np.asarray(x, dtype=int)
    alpha = arr.sum() + 0.5
    beta = (n - arr).sum() + 0.5
    if percentiles is None:
        lower, mean, upper = scipy.special.btdtri(
            alpha, beta, [0.025, 0.5, 0.975])
        stats = pd.Series(
            [lower, mean, upper],
            index=['lower', 'median', 'upper'],
            name=x.name)
    else:
        stats = pd.Series(
            scipy.special.btdtri(alpha, beta, percentiles),
            index=percentiles,
            name=x.name)
    return stats

def bootcorr(x, y, nsamples=10000, method='pearson'):
    arr1 = np.asarray(x)
    arr2 = np.asarray(y)
    n, = arr1.shape
    assert arr1.shape == arr2.shape

    boot_corr = np.empty(nsamples)

    for i in xrange(nsamples):
        boot_idx = np.random.randint(0, n, n)
        boot_arr1 = arr1[boot_idx]
        boot_arr2 = arr2[boot_idx]

        if method == 'pearson':
            func = scipy.stats.pearsonr
        elif method == 'spearman':
            func = scipy.stats.spearmanr
        else:
            raise ValueError("invalid method: %s" % method)

        ii = ~np.isnan(boot_arr1) & ~np.isnan(boot_arr2)
        boot_corr[i] = func(boot_arr1[ii], boot_arr2[ii])[0]

    stats = pd.Series(
        np.percentile(boot_corr, [2.5, 50, 97.5]),
        index=['lower', 'median', 'upper'])

    return stats
