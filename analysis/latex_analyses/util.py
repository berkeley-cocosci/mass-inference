from argparse import ArgumentParser, RawTextHelpFormatter

import os
import json
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def load_config():
    with open(os.path.join(ROOT, "config.json"), "r") as fh:
        config = json.load(fh)
    return config


def newcommand(name, val):
    fmt = load_config()["latex"]["newcommand"]
    return fmt.format(name=name, action=val)


def default_argparser(module):
    config = load_config()

    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    dest = os.path.join(ROOT, config["paths"]["latex"], name + ".tex")
    results_path = os.path.join(ROOT, config["paths"]["results"])
    depends = "\n    ".join(
        ["RESULTS_PATH/{}".format(x) for x in module['__depends__']])

    parser = ArgumentParser(
        description="{}\n\nDependencies:\n\n    {}".format(module['__doc__'], depends),
        formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '--to',
        default=dest,
        help='where to save the resulting latex file\ndefault: %(default)s')
    parser.add_argument(
        '--results-path',
        default=results_path,
        help='directory where the results are located\ndefault: %(default)s')

    return parser



