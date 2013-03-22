import os
import subprocess
import shutil
import pickle

from optparse import OptionParser

cmd_template = "ffmpeg -loglevel error -i %s -r 30 -b 2048k -s 640x480 %s"

VIDEOPATH = "../../stimuli/www/%s/video"
IMAGEPATH = "../../stimuli/www/%s/images"
RENDERPATH = "../../stimuli/render"
CONFPATH = "../../stimuli/meta"


def parseStimParams(stim):
    """Parse the stimulus name into the base name and other parameters
    (e.g., kappa, counterbalance)

    """

    parts = stim.split("~")
    basename = parts[0]
    if len(parts) == 2:
        strparams = parts[1]
        params = tuple([tuple(x.split("-", 1)) for x in strparams.split("_")])
    elif len(parts) > 2:
        raise ValueError("malformed stimulus name: %s" % stim)
    else:
        params = ()
    return basename, set(params)


def getCondition(stim):
    """Condense the parameters down into a shorter, standardized
    condition name.

    """

    basename, params = parseStimParams(stim)
    if len(params) > 0:
        condition = []
        dparams = dict(params)
        if 'kappa' in dparams:
            num = 10 ** float(dparams['kappa'])
            if num == int(num):
                num = int(num)
            condition.append("%s" % num)
        if 'cb' in dparams:
            condition.append("cb%s" % dparams['cb'])
        condition = "-".join(condition)
    else:
        condition = None
    return condition


def parseFiles(path):
    """Group all files in 'path' by base stimulus. Returns a
    dictionary indexed by the base stimulus, where the values are
    a list of tuples consisting of:

        suffix: e.g. feedback, floor
        imgid: e.g. A, B (first frame/last frame)
        ext: extension
        f: full original file name

    For each file matching the base stimulus.

    """

    files = [x for x in os.listdir(path)]
    suffixes = ['feedback', 'floor']
    stims = {}

    for f in files:
        # get rid of the extension
        sf, ext = os.path.splitext(f)
        # parse the filename into parts
        parts = sf.split("~")
        if len(parts[-1]) == 1:
            imgid = parts[-1]
            parts = parts[:-1]
        else:
            imgid = None
        key = "~".join(parts)

        # remove the suffix, if any
        suffix = None
        last = key.split("-")[-1]
        if last in suffixes:
            suffix = last
            key = key[:-(len(suffix)+1)]

        # parse the parameters
        if key not in stims:
            stims[key] = []
        stims[key].append((suffix, imgid, ext, f))

    return stims


def convertFiles(newname, matchinfo, target, formats, dryrun=False):
    """Convert media files specified in `matchinfo` that correspond to
    the base stimulus given by `newname`. Files are saved into a
    folder denoted by a combination of `VIDEOPATH` or `IMAGEPATH` and
    `target`. Image files are merely copied, while video files are
    converted to the formats given in `formats`.

    """

    # paths
    render_path = os.path.join(RENDERPATH, target)
    video_path = VIDEOPATH % target
    image_path = IMAGEPATH % target

    # create the folders if they don't exist
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # for each of the files we need to convert...
    for fileinfo in matchinfo:
        # figure out the new file name
        suffix, imgid, ext, filename = fileinfo
        inpath = os.path.join(render_path, filename)
        if suffix == "feedback":
            suffix = "fb"
        newsuffix = (suffix if suffix else "") + (imgid if imgid else "")
        if newsuffix != "":
            newsuffix = "~" + newsuffix

        if ext == ".png":
            # copy if the file is an image
            outname = newname + newsuffix + ext
            outpath = os.path.join(image_path, outname)
            if os.path.exists(outpath):
                print "    %s exists" % outname
                continue
            else:
                print "    %s --> %s" % (filename, outname)

            if not dryrun:
                shutil.copy(inpath, outpath)

        elif ext == ".avi":
            # convert if the file is a video
            outnamebase = newname + newsuffix
            for fmt in formats:
                outname = outnamebase + "." + fmt
                outpath = os.path.join(video_path, outname)
                if os.path.exists(outpath):
                    print "    %s exists" % outname
                    continue
                else:
                    print "    %s --> %s" % (filename, outname)

                if not dryrun:
                    cmd = cmd_template % (inpath, outpath)
                    subprocess.call(cmd, shell=True)


def convert(target, formats, dryrun):
    if dryrun:
        print "DRYRUN -- not actually making changes"

    # load stimulus info
    conf_path = os.path.join(CONFPATH, "%s-stimulus-info.pkl" % target)
    with open(conf_path, "r") as fh:
        stiminfo = pickle.load(fh)

    scenes = sorted(stiminfo.keys())
    convinfo = {}
    convinfo_inv = {}

    # convert scene names
    for scene in scenes:
        info = stiminfo[scene]
        if info['newname'] is not None:
            newname = info['newname']
        else:
            newname, params = parseStimParams(scene)
        condition = getCondition(scene)
        suffix = ("~%s" % condition) if condition else ""
        newscene = newname + suffix
        convinfo[newscene] = scene
        convinfo_inv[scene] = newscene

    # save conversions to file
    if not dryrun:
        conv_table = os.path.join(CONFPATH, '%s-conversion.pkl' % target)
        with open(conv_table, 'w') as fh:
            pickle.dump(convinfo, fh)
        print "Saved conversion information to '%s'." % conv_table

    # parse the existing filenames, to figure out which files we need
    # to convert
    render_path = os.path.join(RENDERPATH, target)
    fp = parseFiles(render_path)

    # convert the actual files
    for sidx, scene in enumerate(scenes):
        files = fp[scene]
        newscene = convinfo_inv[scene]

        print "="*70
        print "[%d/%d] %s --> %s" % (sidx+1, len(scenes), scene, newscene)
        print "-"*70

        convertFiles(newscene, files, target, formats, dryrun=dryrun)

###############

if __name__ == "__main__":
    usage = "usage: %prog [options] target"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--mp4", dest="use_mp4", action="store_true",
        default=False, help="convert to mp4 format")
    parser.add_option(
        "--ogg", dest="use_ogg", action="store_true",
        default=False, help="convert to ogg format")
    parser.add_option(
        "--flv", dest="use_flv", action="store_true",
        default=False, help="convert to flv format")
    parser.add_option(
        "--wmv", dest="use_wmv", action="store_true",
        default=False, help="convert to wmv format")
    parser.add_option(
        "--dry-run",
        action="store_true", dest="dryrun", default=False,
        help="do not actually do anything")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        raise ValueError("no target directory name specified")
    else:
        target = args[0]

    outext = []
    if options.use_mp4:
        outext.append("mp4")
    if options.use_ogg:
        outext.append("ogg")
    if options.use_flv:
        outext.append("flv")
    if options.use_wmv:
        outext.append("wmv")
    if len(outext) == 0:
        print "** WARNING: No conversion formats specified! **"

    dryrun = options.dryrun

    convert(target, outext, dryrun)
