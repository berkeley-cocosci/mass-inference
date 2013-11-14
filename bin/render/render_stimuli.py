#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass import RENDER_SCRIPT_PATH as SCRIPT_PATH
from mass.render import tasks
import logging
import multiprocessing
import sys

logger = logging.getLogger("mass.render")


def make_parser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-c", "--condition",
        default="*",
        help=("Name of the condition. If not provided, "
              "all conditions will be rendered."))
    parser.add_argument(
        "-t", "--tag",
        default="*",
        help=("Name of the stimulus set. If not provided, "
              "all stimulus sets will be rendered."))
    parser.add_argument(
        "-f", "--force",
        default=False,
        action="store_true",
        help="Force all tasks to be put on the queue.")

    encopt = parser.add_argument_group(title="Encoding options")
    encopt.add_argument(
        "--fps",
        type=float,
        default=30,
        help="frames per second")
    encopt.add_argument(
        "--ext",
        default="png",
        choices=["png", "jpeg"],
        help="file format to save frames as")

    return parser


def render(script, options):
    from mass.render.renderer import render
    render(script, **options)


if __name__ == "__main__":
    parser = make_parser()
    args = vars(parser.parse_args())

    # Extract render options
    options = {
        'fps': args['fps'],
        'ext': args['ext'],
        'force': args['force'],
    }

    root = SCRIPT_PATH.joinpath(args['exp'])
    script_paths = root.glob("%s/%s.json" % (args['condition'], args['tag']))
    script = tasks.load_tasks(script_paths)

    # Run the renderer
    p = multiprocessing.Process(target=render, args=(script, options))
    p.start()
    p.join()

    if p.exitcode != 0:
        logger.error("Render exited with code %d", p.exitcode)
        sys.exit(p.exitcode)
