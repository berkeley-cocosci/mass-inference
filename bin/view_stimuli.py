#!/usr/bin/env python

import os
import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(root, "lib"))

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass.stimuli import get_style
from mass.render.viewer import init
from path import path


def parseargs():

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "stimuli", metavar="stim", type=str, nargs="+",
        help="path to stimulus")
    parser.add_argument(
        "-k", "--kappa", dest="kappa",
        action="store", type=float, default=0.0,
        help=("log10 mass ratio of type1:type0 blocks, "
              "only for mass towers (default: 0.0)"))
    parser.add_argument(
        "--camera-start",
        action="store", dest="camstart", type=int,
        help="initial camera angle")
    parser.add_argument(
        "--color0",
        action="store", dest="color0", type=str, default=None,
        help="color of type 0 blocks")
    parser.add_argument(
        "--color1",
        action="store", dest="color1", type=str, default=None,
        help="color of type 1 blocks")

    args = parser.parse_args()
    stims = [path(x).abspath() for x in args.stimuli]
    stimtypes = [get_style(pth.dirname().name) for pth in stims]

    N = len(stims)
    opts = {
        'stimulus': stims,
        'stimtype': stimtypes,
        'kappa': [args.kappa]*N,
        'camera_start': [args.camstart]*N,
        'color0': [args.color0]*N,
        'color1': [args.color1]*N
    }
    return opts


if __name__ == "__main__":
    opts = parseargs()
    init(opts)
