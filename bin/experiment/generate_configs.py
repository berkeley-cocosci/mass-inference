#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import dbtools
import json
import numpy as np
import pandas as pd
from path import path
from mass import EXP_PATH, CPO_PATH
from mass import RENDER_SCRIPT_PATH as SCRIPT_PATH
from itertools import product
import logging
from termcolor import colored

logger = logging.getLogger("mass.experiment")

conditions = {
    "0": {
        "pretest": "shared",
        "experimentA": "vfb-0.1",
        "experimentB": "nfb-0.1",
        "experimentC": "vfb-0.1",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-0.1"
    },

    "1": {
        "pretest": "shared",
        "experimentA": "vfb-0.1",
        "experimentB": "nfb-0.1",
        "experimentC": "vfb-10",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-0.1"
    },

    "2": {
        "pretest": "shared",
        "experimentA": "vfb-10",
        "experimentB": "nfb-10",
        "experimentC": "vfb-0.1",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-10"
    },

    "3": {
        "pretest": "shared",
        "experimentA": "vfb-10",
        "experimentB": "nfb-10",
        "experimentC": "vfb-10",
        "posttest": "shared",
        "unstable_example": "shared",
        "stable_example": "shared",
        "mass_example": "vfb-10"
    },
}


def load_script(exp, cond, phase, cb):
    # pretest is the same as the posttest
    if phase == "posttest":
        phase = "pretest"

    # stimuli in shared aren't counterbalanced
    if cond == "shared":
        render_script = SCRIPT_PATH.joinpath(
            exp, cond, "%s.json" % phase)

    else:
        render_script = SCRIPT_PATH.joinpath(
            exp, "%s-cb%d" % (cond, cb), "%s.json" % phase)

    with open(render_script, "r") as fh:
        script = json.load(fh)

    script = pd.DataFrame\
               .from_dict(script)\
               .drop(labels=["render_root", "finished"], axis=1)
    script.stimulus = script.stimulus.map(lambda x: path(x).namebase)

    return script


def parse_cond(cond, phase):
    if phase in ("pretest", "posttest"):
        fb = "vfb"
        ratio = "1"

    elif phase in ("stable_example", "unstable_example"):
        fb = "vfb"
        ratio = "1"

    # XXX: hack! ratio might not be correct for the mass example
    # because we want the example tower to be the same for
    # everyone (including whether it falls or not), so we don't
    # actually use a different kappa
    elif phase == "mass_example":
        fb = "vfb"
        ratio = "10"

    else:
        fb, ratio = cond.split("-")

    return fb, ratio


def load_meta(tbl, stim, kappa):
    meta = tbl.select(where=("stimulus=? and kappa=?", (stim, kappa)))
    if len(meta) == 0:
        raise RuntimeError("could not find metadata for '%s' with k=%s" % (
            stim, kappa))
    if len(meta) > 1:
        if not (meta['nfell'][0] == meta['nfell']).all():
            logger.warning("nfell for stim '%s' is different "
                           "for different datasets!" % stim)

        if not (meta['stable'][0] == meta['stable']).all():
            logger.error(colored("stable for stim '%s' is different "
                                 "for different datasets!" % stim, 'red'))
            print meta

    meta['stable'] = meta['stable'].astype('bool')
    meta = meta.drop('dataset', axis=1)[:1]
    return meta


def gen_config(exp, cond, phase, cb):
    fb, ratio = parse_cond(cond, phase)

    # load render script
    script = load_script(exp, cond, phase, cb)
    script['feedback'] = fb
    script['ratio'] = ratio
    script['counterbalance'] = cb

    # load metadata (e.g. stability)
    tbl = dbtools.Table(CPO_PATH.joinpath("metadata.db"), "stability")
    meta = pd.concat(map(
        lambda s, k: load_meta(tbl, s, k),
        script.stimulus, script.kappa))

    # merge metadata and render script to get the config
    config = pd.merge(script, meta, on=['stimulus', 'kappa'])
    config = config.set_index('stimulus').sort().reset_index()

    # sanity check, to make sure ratios match kappas
    r2kappa = np.array(map(float, config.ratio))
    assert (10**config.kappa == r2kappa).all()

    # dump dataframe to a dictionary and sort by stimulus
    trials = config.reset_index(drop=True).T.to_dict().values()
    trials.sort(cmp=lambda x, y: cmp(x['stimulus'], y['stimulus']))

    # make sure there are no nans, because javascript won't load JSON
    # with nans in it
    for trial in trials:
        for key, val in trial.iteritems():
            if isinstance(val, float) and np.isnan(val):
                if key in ("color0", "color1"):
                    trial[key] = None

    # only return a list if there are multiple trials
    if len(trials) == 1:
        trials = trials[0]

    return trials


def save_conditions(force=False):
    cond_path = EXP_PATH.joinpath(
        "static", "json", "conditions.json").abspath()
    if cond_path.exists() and not force:
        return
    if not cond_path.dirname().exists():
        cond_path.dirname().makedirs_p()
    with open(cond_path, "w") as fh:
        json.dump(conditions, fh, indent=2, allow_nan=False)

    logger.info("Saved %s", cond_path.relpath())


def save_config(exp, condition_num, cb, force=False):
    config_path = EXP_PATH.joinpath(
        "static", "json", "%s-cb%d.json" % (condition_num, cb)).abspath()

    if config_path.exists() and not force:
        return

    config = {}
    phases = sorted(conditions[condition_num].keys())
    for phase in phases:
        cond = conditions[condition_num][phase]
        config[phase] = gen_config(exp, cond, phase, cb)

    if not config_path.dirname().exists():
        config_path.dirname().makedirs_p()

    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2, allow_nan=False)

    logger.info("Saved %s", config_path.relpath())


def make_parser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force all tasks to be put on the queue.")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    save_conditions(force=args.force)
    conds = sorted(conditions.keys())
    for cond_num, cb in product(conds, [0, 1]):
        save_config(args.exp, cond_num, cb, force=args.force)
