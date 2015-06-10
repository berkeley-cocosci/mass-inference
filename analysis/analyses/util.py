import os
import numpy as np
import sys
import json
import pandas as pd
import scipy.special
import scipy.stats

from argparse import ArgumentParser, RawTextHelpFormatter
import datapackage as dpkg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MAX_LOG = np.log(sys.float_info.max)


def bootstrap_mean(x, nsamples=10000):
    arr = np.asarray(x)
    n, = arr.shape
    boot_idx = np.random.randint(0, n, n * nsamples)
    boot_arr = arr[boot_idx].reshape((n, nsamples))
    boot_mean = boot_arr.mean(axis=0)
    lower, median, upper = np.percentile(boot_mean, [2.5, 50, 97.5])
    stats = pd.Series(
        [n, lower, median, upper],
        index=['N', 'lower', 'median', 'upper'],
        name=x.name)
    return stats


def beta(x, n=1, percentiles=None):
    arr = np.asarray(x, dtype=int)
    alpha = arr.sum() + 0.5
    beta = (n - arr).sum() + 0.5
    if percentiles is None:
        lower, mean, upper = scipy.special.btdtri(
            alpha, beta, [0.025, 0.5, 0.975])
        stats = pd.Series(
            [arr.size, lower, mean, upper],
            index=['N', 'lower', 'median', 'upper'],
            name=x.name)
    else:
        stats = pd.Series(
            scipy.special.btdtri(alpha, beta, percentiles),
            index=percentiles,
            name=x.name)
    return stats


def beta_stddev(x):
    arr = np.asarray(x, dtype=int)
    alpha = arr.sum() + 0.5
    beta = (1 - arr).sum() + 0.5
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    stddev = np.sqrt(var)
    return stddev


def beta_mean(x):
    arr = np.asarray(x, dtype=int)
    alpha = arr.sum() + 0.5
    beta = (1 - arr).sum() + 0.5
    mean = alpha / (alpha + beta)
    return mean


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


def normalize(logarr, axis=-1):
    """Normalize an array of log-values.

    This function is very useful if you have an array of log
    probabilities that need to be normalized, but some of the
    probabilies might be extremely small (i.e., underflow will occur if
    you try to exponentiate them). This function computes the
    normalization constants in log space, thus avoiding the need to
    exponentiate the values.

    Parameters
    ----------
    logarr: numpy.ndarray
        Array of log values
    axis: integer (default=-1)
        Axis over which to normalize

    Returns
    -------
    out: (numpy.ndarray, numpy.ndarray)
        2-tuple consisting of the log normalization constants used to
        normalize the array, and the normalized array of log values

    """

    # shape for the normalization constants (that would otherwise be
    # missing axis)
    shape = list(logarr.shape)
    shape[axis] = 1
    # get maximum value of array
    maxlogarr = logarr.max(axis=axis).reshape(shape)
    # calculate how much to shift the array up by
    shift = MAX_LOG - maxlogarr - 2 - logarr.shape[axis]
    shift[shift < 0] = 0
    # shift the array
    unnormed = logarr + shift
    # convert from logspace
    arr = np.exp(unnormed)
    # calculate shifted log normalization constants
    _lognormconsts = np.log(arr.sum(axis=axis)).reshape(shape)
    # calculate normalized array
    lognormarr = unnormed - _lognormconsts
    # unshift normalization constants
    _lognormconsts -= shift
    # get rid of the dimension we normalized over
    lognormconsts = _lognormconsts.sum(axis=axis)

    return lognormconsts, lognormarr


def load_config():
    with open(os.path.join(ROOT, "config.json"), "r") as fh:
        config = json.load(fh)
    return config


def load_human(data_path):
    config = load_config()
    version = config["analysis"]["human_version"]

    exp_dp = dpkg.DataPackage.load(os.path.join(
        data_path, "human/mass_inference-%s.dpkg" % version))
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


def load_participants(data_path):
    config = load_config()
    version = config["analysis"]["human_version"]
    exp_dp = dpkg.DataPackage.load(os.path.join(
        data_path, "human/mass_inference-%s.dpkg" % version))
    participants = exp_dp.load_resource("participants.csv")
    participants = participants.reset_index()
    return participants


def load_ipe(data_path):
    def load(version, block):
        path = os.path.join(
            data_path, "model/mass_inference-%s-%s_ipe_fall.dpkg" % (version, block.lower()))
        dp = dpkg.DataPackage.load(path)
        data = dp.load_resource("model.csv")
        data["block"] = block
        return data

    config = load_config()
    version = config["analysis"]["model_version"]
    ipe = pd.concat([
        load(version, "A"),
        load(version, "B")
    ])

    return ipe


def load_fb(data_path):
    config = load_config()
    version = config["analysis"]["model_version"]

    def load(name):
        path = os.path.join(data_path, "model/%s.dpkg" % name)
        dp = dpkg.DataPackage.load(path)
        data = dp.load_resource("model.csv").set_index(['sigma', 'phi', 'stimulus'])
        return data

    def load_fb(name):
        data = load(name)
        return data\
            .reset_index()\
            .groupby(['sigma', 'phi', 'sample'])\
            .get_group((0.0, 0.0, 0))\
            .drop(['sigma', 'phi', 'sample'], axis=1)\
            .set_index(['stimulus', 'kappa'])

    def load_nofb(name):
        data = load_fb(name)
        data[:] = np.nan
        return data

    fb = {
        'A': load_nofb("mass_inference-%s-a_truth_fall" % version),
        'B': load_nofb("mass_inference-%s-b_truth_fall" % version),
        'C': load_fb("mass_inference-%s-b_truth_fall" % version),
    }

    return fb


def get_dependencies(depends, config, data_path="DATA_PATH", results_path="RESULTS_PATH"):
    human_version = config["analysis"]["human_version"]
    model_version = config["analysis"]["model_version"]

    special_deps = {
        'human': "human/mass_inference-{}.dpkg".format(human_version),
        'ipe_A': "model/mass_inference-{}-a_ipe_fall.dpkg".format(model_version),
        'ipe_B': "model/mass_inference-{}-b_ipe_fall.dpkg".format(model_version),
        'fb_A': "model/mass_inference-{}-a_truth_fall.dpkg".format(model_version),
        'fb_B': "model/mass_inference-{}-b_truth_fall.dpkg".format(model_version),
    }

    full_deps = []
    use_results_path = False
    use_data_path = False
    for dep in depends:
        if dep in special_deps:
            full_deps.append("{}/{}".format(data_path, special_deps[dep]))
            use_data_path = True
        else:
            full_deps.append("{}/{}".format(results_path, dep))
            use_results_path = True

    return full_deps, use_data_path, use_results_path


def default_argparser(module):
    config = load_config()

    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    dest = os.path.join(
        ROOT, config["paths"]["results"], name + module.get('__ext__', '.csv'))
    depends, data_path, results_path = get_dependencies(
        module['__depends__'], config)

    if len(depends) > 0:
        description = "{}\n\nDependencies:\n\n    {}".format(
            module['__doc__'], "\n    ".join(depends))
    else:
        description = module['__doc__']

    parser = ArgumentParser(
        description=description,
        formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '--to',
        default=dest,
        help='where to save out the results\ndefault: %(default)s')

    if data_path:
        parser.add_argument(
            '--data-path',
            default=os.path.join(ROOT, config["paths"]["data"]),
            help='where other results are saved\ndefault: %(default)s')

    if results_path:
        parser.add_argument(
            '--results-path',
            default=os.path.join(ROOT, config["paths"]["results"]),
            help='where other results are saved\ndefault: %(default)s')

    if module.get('__random__', False):
        parser.add_argument(
            '--seed', default=config["analysis"]["seed"],
            type=int, help='seed for the random number generator (default: %(default)s)')

    if module.get('__parallel__', False):
        parser.add_argument(
            '--serial',
            dest='parallel',
            action='store_false',
            help="don't run analysis in parallel")

    return parser


def get_params():
    config = load_config()
    sigma = config["analysis"]["sigma"]
    phi = config["analysis"]["phi"]
    return sigma, phi


def get_query():
    config = load_config()
    return config["analysis"]["query"]
