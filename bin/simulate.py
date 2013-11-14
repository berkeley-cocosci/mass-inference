#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass import BIN_PATH
from termcolor import colored
import logging
import subprocess
import sys


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
        "-t", "--tag",
        required=True,
        help="short name indetifying the stimulus set")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="force tasks to complete")

    group = parser.add_argument_group(title="simulate operations")
    group.add_argument(
        "-a", "--all",
        action="store_true",
        default=False,
        help="perform all actions")
    group.add_argument(
        "--generate",
        action="store_true",
        default=False,
        help="generate simulation scripts")
    group.add_argument(
        "--run-server",
        dest="run_server",
        action="store_true",
        default=False,
        help="run simulation server")
    group.add_argument(
        "--run-client",
        dest="run_client",
        action="store_true",
        default=False,
        help="run simulation client")
    group.add_argument(
        "--process",
        action="store_true",
        default=False,
        help="process simulation data")
    group.add_argument(
        "--query",
        action="store_true",
        default=False,
        help="compute model queries from simulation data")
    group.add_argument(
        "--extract",
        action="store_true",
        default=False,
        help="extract query feedback from simulation data")

    args = parser.parse_args()
    required = [
        args.generate,
        args.run_server,
        args.run_client,
        args.process,
        args.query,
        args.extract,
        args.all
    ]

    if not any(required):
        print colored("You must specify at least one action!\n", 'red')
        parser.print_help()
        sys.exit(1)

    exp = args.exp
    tag = args.tag
    force = args.force

    # generate configs
    if args.generate or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("simulate/generate_script.py"),
            "-e", exp, "-t", tag
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # run experiment server
    if args.run_server or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("simulate/run_simulations.py"),
            "server", "-e", exp, "-t", tag, "-k", "zo7MV6GndfNf"
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # run experiment client
    if args.run_client:
        run_cmd([
            "python", BIN_PATH.joinpath("simulate/run_simulations.py"),
            "client", "-k", "zo7MV6GndfNf", "-s"
        ])

    # process simulation data
    if args.process or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("simulate/process_simulations.py"),
            "-e", exp, "-t", tag
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # compute model queries
    if args.query or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("simulate/query_model.py"),
            "-e", exp, "-t", tag
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)

    # compute model queries
    if args.extract or args.all:
        cmd = [
            "python", BIN_PATH.joinpath("simulate/extract_feedback.py"),
            "-e", exp, "-t", tag
        ]
        if force:
            cmd.append("-f")
        run_cmd(cmd)
