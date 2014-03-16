#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from mass import RENDER_PATH
from path import path
import logging
import subprocess

logger = logging.getLogger('mass.render')


def convert(video, formats, force):
    for fmt in formats:
        newfile = path(video).splitext()[0] + "." + fmt
        if force or not newfile.exists():
            cmd = ("ffmpeg -loglevel error -i %s -y "
                   "-r 30 -b 2048k -s 640x480 %s" % (
                       video, newfile))
            logging.info(cmd)
            code = subprocess.call(cmd, shell=True)
            if code != 0:
                raise RuntimeError("conversion exited abnormally: %d" % code)
        else:
            print "exists: %s" % newfile.relpath()


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
        "-f", "--force",
        default=False,
        action="store_true",
        help="Force all videos to be converted.")

    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    video_path = RENDER_PATH.joinpath(args.exp)
    videos = video_path.glob("%s/*.avi" % args.condition)
    for video in videos:
        convert(video, ['mp4', 'ogg', 'webm'], force=args.force)
