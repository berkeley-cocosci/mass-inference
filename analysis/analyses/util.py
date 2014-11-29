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


def load_config(root):
    with open(os.path.join(root, "config.json"), "r") as fh:
        config = json.load(fh)
    return config


def load_human():
    root = ".."
    config = load_config(root)
    human_version = config["analysis"]["human_version"]
    data_path = os.path.abspath(os.path.join(
        root, config["analysis"]["data_path"]))
    human = _load_human(human_version, data_path)
    return human


def load_participants():
    root = ".."
    config = load_config(root)
    human_version = config["analysis"]["human_version"]
    data_path = os.path.abspath(os.path.join(
        root, config["analysis"]["data_path"]))
    human = _load_participants(human_version, data_path)
    return human


def load_ipe():
    root = ".."
    config = load_config(root)
    model_version = config["analysis"]["model_version"]
    data_path = os.path.abspath(os.path.join(
        root, config["analysis"]["data_path"]))
    ipe = _load_ipe(model_version, data_path)
    return ipe


def load_fb():
    root = ".."
    config = load_config(root)
    model_version = config["analysis"]["model_version"]
    data_path = os.path.abspath(os.path.join(
        root, config["analysis"]["data_path"]))
    fb = _load_fb(model_version, data_path)
    return fb


def load_model():
    ipe = load_ipe()
    fb = load_fb()
    return ipe, fb


def load_all():
    root = ".."
    config = load_config(root)
    human_version = config["analysis"]["human_version"]
    model_version = config["analysis"]["model_version"]
    data_path = os.path.abspath(os.path.join(
        root, config["analysis"]["data_path"]))
    data = _load_all(
        model_version=model_version,
        human_version=human_version,
        data_path=data_path)
    return data


def default_argparser(doc, results_path=False, seed=False, parallel=False):
    root = ".."
    config = load_config(root)

    parser = ArgumentParser(description=doc, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        'dest', help='where to save out the results')

    if results_path:
        parser.add_argument(
            '-r', '--results-path',
            default=os.path.abspath(os.path.join(
                root, config["analysis"]["results_path"])),
            help='where other results are saved\ndefault: %(default)s')

    if seed:
        parser.add_argument(
            '-s', '--seed', default=config["analysis"]["seed"],
            type=int, help='seed for the random number generator (default: %(default)s)')

    if parallel:
        parser.add_argument(
            '--serial',
            dest='parallel',
            action='store_false',
            help="don't run analysis in parallel")

    return parser


def get_params():
    config = load_config("..")
    sigma = config["analysis"]["sigma"]
    phi = config["analysis"]["phi"]
    return sigma, phi


def get_query():
    config = load_config("..")
    query = config["analysis"]["query"]
    return query
