from argparse import ArgumentParser, RawTextHelpFormatter
import os
import numpy as np
import sys
import json

from mass.analysis import load_human as _load_human
from mass.analysis import load_ipe as _load_ipe
from mass.analysis import load_fb as _load_fb
from mass.analysis import load_all as _load_all
from mass.analysis import load_participants as _load_participants
from mass.analysis import bootcorr, bootstrap_mean, beta


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MAX_LOG = np.log(sys.float_info.max)


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
    human_version = config["analysis"]["human_version"]
    human = _load_human(human_version, data_path)
    return human


def load_participants(data_path):
    config = load_config()
    human_version = config["analysis"]["human_version"]
    human = _load_participants(human_version, data_path)
    return human


def load_ipe(data_path):
    config = load_config()
    model_version = config["analysis"]["model_version"]
    ipe = _load_ipe(model_version, data_path)
    return ipe


def load_fb(data_path):
    config = load_config()
    model_version = config["analysis"]["model_version"]
    fb = _load_fb(model_version, data_path)
    return fb


def load_model(data_path):
    ipe = load_ipe(data_path)
    fb = load_fb(data_path)
    return ipe, fb


def load_all(data_path):
    config = load_config()
    human_version = config["analysis"]["human_version"]
    model_version = config["analysis"]["model_version"]
    data = _load_all(
        model_version=model_version,
        human_version=human_version,
        data_path=data_path)
    return data


def get_dependencies(depends, config):
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
            full_deps.append("DATA_PATH/{}".format(special_deps[dep]))
            use_data_path = True
        else:
            full_deps.append("RESULTS_PATH/{}".format(dep))
            use_results_path = True

    return full_deps, use_data_path, use_results_path

def default_argparser(module, seed=False, parallel=False, ext=".csv"):
    config = load_config()

    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    dest = os.path.join(ROOT, config["paths"]["results"], name + ext)
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
        '--dest',
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

    if seed:
        parser.add_argument(
            '--seed', default=config["analysis"]["seed"],
            type=int, help='seed for the random number generator (default: %(default)s)')

    if parallel:
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
