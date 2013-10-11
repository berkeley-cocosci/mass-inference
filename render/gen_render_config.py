from argparse import ArgumentParser
from path import path
from view_towers import STIMTYPES
import numpy as np
import pandas as pd


def gen_angles(n):
    angles = np.random.randint(0, 360, n)
    return angles


def parseargs():

    parser = ArgumentParser()
    parser.add_argument(
        "stimuli", metavar="stim", type=str, nargs="+",
        help="path to stimulus")
    parser.add_argument(
        "-o", "--out", dest="filename", type=str, action="store",
        help="output filename", required=True)
    parser.add_argument(
        "-s", "--stype", dest="stype", action="store",
        help=("stimulus type. If not provided, it will be inferred "
              "from the paths to the stimuli."),
        choices=sorted(set(STIMTYPES.values())))
    parser.add_argument(
        "-k", "--kappa", dest="kappa",
        action="store", type=float, default=0.0,
        help="log10 mass ratio, only for mass towers (default: 0.0)")
    parser.add_argument(
        "--presentation-time",
        action="store", dest="ptime", type=float, default=5.0,
        help=("amount of time in seconds to present the stimulus "
              "for (default: 5.0)"))
    parser.add_argument(
        "--feedback-time",
        action="store", dest="ftime", type=float, default=2.5,
        help="amount of time in seconds to show feedback for (default: 2.5)")
    parser.add_argument(
        "--label0",
        action="store", dest="label0", type=float, default="0",
        help="label of type 0 blocks")
    parser.add_argument(
        "--label1",
        action="store", dest="label1", type=float, default="1",
        help="label of type 1 blocks")
    parser.add_argument(
        "--color0",
        action="store", dest="color0", type=float, default=None,
        help="color of type 0 blocks")
    parser.add_argument(
        "--color1",
        action="store", dest="color0", type=float, default=None,
        help="color of type 1 blocks")
    parser.add_argument(
        "--flip-colors",
        action="store_true", dest="flip", default=False,
        help="swap block colors")
    parser.add_argument(
        "--no-feedback",
        action="store_false", dest="feedback", default=True,
        help="do not render feedback")
    parser.add_argument(
        "--occlude",
        action="store_true", dest="occlude", default=False,
        help="drop occluder after stimulus presentation")
    parser.add_argument(
        "--full",
        action="store_true", dest="full_render", default=False,
        help="render the full stimulus as one video")

    args = parser.parse_args()
    stims = [path(x).abspath() for x in args.stimuli]
    if args.stype:
        stimtypes = [args.stype]*len(stims)
    else:
        stimtypes = [STIMTYPES[x.splitall()[-2]] for x in stims]

    N = len(stims)
    options = {
        'stimulus': stims,
        'stimtype': stimtypes,
        'kappa': [args.kappa]*N,
        'flip_colors': [args.flip]*N,
        'feedback': [args.feedback]*N,
        'occlude': [args.occlude]*N,
        'presentation_time': [args.ptime]*N,
        'feedback_time': [args.ftime]*N,
        'angle': gen_angles(N),
        'full_render': [args.full_render]*N,
        'color0': [args.color0]*N,
        'color1': [args.color1]*N,
        'label0': [args.label0]*N,
        'label1': [args.label1]*N
    }

    return options, args.filename


def save_options(options, filename):
    df = pd.DataFrame(options).set_index('stimulus')
    pth = path(filename).splitpath()[0]
    if not pth.exists():
        pth.makedirs_p()
    df.to_csv(filename)


if __name__ == "__main__":
    options, filename = parseargs()
    save_options(options, filename)
