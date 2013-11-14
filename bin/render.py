#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass import RENDER_PATH, BIN_PATH
from mass import RENDER_SCRIPT_PATH as SCRIPT_PATH
from termcolor import colored
import logging
import subprocess
import sys

logger = logging.getLogger('mass.render')


def run_cmd(cmd):
    logging.info(colored("Running %s" % " ".join(cmd), 'blue'))
    code = subprocess.call(cmd)
    if code != 0:
        raise RuntimeError("Process exited abnormally: %d" % code)


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="experiment version")
    parser.add_argument(
        "-c", "--condition",
        default="*",
        help="condition name")
    parser.add_argument(
        "-t", "--tag",
        default="*",
        help="short name indetifying the stimulus set")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="force tasks to complete")
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="remove render/script directories first")

    group = parser.add_argument_group(title="render operations")
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=False,
        help="perform all actions")
    group.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="generate render scripts")
    group.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="render stimuli")
    group.add_argument(
        "--convert",
        action="store_true",
        default=False,
        help="convert videos")

    args = parser.parse_args()
    required = [
        args.generate,
        args.delete,
        args.render,
        args.convert,
        args.all
    ]

    if not any(required):
        print colored("You must specify at least one action!\n", 'red')
        parser.print_help()
        sys.exit(1)

    exp = args.exp
    cond = args.condition
    tag = args.tag
    force = args.force

    # remove old files
    if args.delete:
        if cond == "*":
            dirs = [SCRIPT_PATH.joinpath(exp),
                    RENDER_PATH.joinpath(exp)]
        else:
            dirs = [SCRIPT_PATH.joinpath(exp, cond),
                    RENDER_PATH.joinpath(exp, cond)]

        for dirname in dirs:
            if not dirname.exists():
                continue
            logging.info("Removing %s", dirname.relpath())
            dirname.rmtree()

    # generate configs
    if args.generate or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("render/generate_script.py"),
            "-e", exp, "-c", cond, "-t", tag
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # render stimuli
    if args.render or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("render/render_stimuli.py"),
            "-e", exp, "-c", cond, "-t", tag
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # convert videos
    if args.convert or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("render/convert_videos.py"),
            "-e", exp, "-c", cond
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)
