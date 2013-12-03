#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass import RENDER_PATH, EXP_PATH
import logging

logger = logging.getLogger('mass.render')


def make_parser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    expdir = EXP_PATH.joinpath("static/stimuli")
    if expdir.exists():
        logger.info("Deleting %s", expdir.relpath())
        expdir.rmtree()
    logger.info("Creating %s", expdir.relpath())
    expdir.makedirs_p()

    stimuli_path = RENDER_PATH.joinpath(args.exp)
    files = stimuli_path.glob("*/*")
    for oldpath in files:
        condition, filename = oldpath.splitall()[-2:]
        if condition == "shared":
            newpaths = [
                expdir.joinpath(condition + "-cb0", filename),
                expdir.joinpath(condition + "-cb1", filename)
            ]
        else:
            newpaths = [expdir.joinpath(condition, filename)]

        for newpath in newpaths:
            directory = newpath.dirname()
            if not directory.exists():
                logging.info("Linking %s --> %s",
                             oldpath.dirname(), directory.relpath())
                directory.makedirs_p()
            oldpath.link(newpath)
