import os
import sys

from optparse import OptionParser


def parseargs():

    usage = ("usage: %prog [options] stimlist\n"
             "       %prog [options] stim1 ... stimN")
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-s", "--stype", dest="stype",
        help="stimulus type, e.g. original [required]",
        metavar="STIM_TYPE")
    parser.add_option(
        "-c", "--camstart", dest="cam_start",
        action="store", type="int", default=-10,
        help="initial camera angle",
        metavar="ANGLE")
    parser.add_option(
        "-k", "--kappa", dest="kappa",
        action="store", type="float", default=None,
        help="log10 mass ratio (only for mass towers)",
        metavar="LOG_RATIO")
    parser.add_option(
        "-p", "--playback", dest="playback",
        action="store_false", default=True,
        help="use playback files instead of computing physics")

    (options, args) = parser.parse_args()

    if options.stype is None:
        print "No stimulus type provided, exiting."
        sys.exit(1)

    cpopath = os.path.join("../../stimuli/obj/old", options.stype)
    listpath = "../../stimuli/lists"

    if len(args) == 0:
        print "No stimuli specified. Loading all stimuli from '%s'" % cpopath
        scenes = os.listdir(cpopath)
        scenes = [x for x in scenes if x != 'name_table.pkl']

    elif len(args) > 1:
        print "Loading scenes from arguments"
        scenes = args

    else:
        lp = os.path.join(listpath, args[0])
        if not os.path.exists(lp):
            print "Loading scene from argument"
            scenes = args
        else:
            print "Loading scenes from list '%s'" % lp
            with open(lp, "r") as fh:
                scenes = fh.read().split("\n")
            scenes = [x for x in scenes if x != '']

    if options.kappa is not None:
        print "Note: overriding stimuli mass ratios"
        newscenes = []
        for scene in scenes:
            if "~" not in scene:
                newscenes.append("%s~kappa-%.1f" % (scene, options.kappa))
            else:
                cponame, strparams = x.split("~")
                pdict = dict([x.split("-", 1) for x in strparams.split("_")])
                pdict[options.kappa] = "%.1f" % options.kappa
                strparams = "_".join(["%s-%s" % (x, pdict[x]) for x in pdict])
                newscene = "%s~%s" % (cponame, strparams)
                newscenes.append(newscene)
        scenes = newscenes

    return scenes, cpopath, options.cam_start, options.playback
