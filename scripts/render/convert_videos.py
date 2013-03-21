import os
import subprocess
import shutil
import json
import numpy as np

from optparse import OptionParser

cmd_template = "ffmpeg -loglevel error -i %s -r 30 -b 2048k -s 640x480 %s"

VIDEOPATH = "../../stimuli/www/%s/video"
IMAGEPATH = "../../stimuli/www/%s/images"
RENDERPATH = "../../stimuli/render"
CONFPATH = "../../stimuli/meta"
LISTPATH = "../../stimuli/lists"


def parseStimParams(stim):
    parts = stim.split("~")
    basename = parts[0]
    if len(parts) == 2:
        strparams = parts[1]
    elif len(parts) > 2:
        raise ValueError("malformed stimulus name: %s" % stim)
    else:
        strparams = ""
    params = tuple([tuple(x.split("-", 1)) for x in strparams.split("_")])
    return basename, set(params)


def compareStims(stim1, stim2):
    base1, params1 = parseStimParams(stim1)
    base2, params2 = parseStimParams(stim2)
    if base1 != base2:
        return False
    issubset = params1 <= params2
    return issubset


def parseFiles(path):
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


def matchStims(stims, path):
    pf = parseFiles(path)
    match = {}
    for stim in stims:
        match[stim] = {}
        for f in pf:
            if compareStims(stim, f):
                match[stim][f] = pf[f]
    return match


def convertStim(stim, newstim, matchinfo, target, formats, dryrun=False):
    render_path = os.path.join(RENDERPATH, target)
    video_path = VIDEOPATH % target
    image_path = IMAGEPATH % target

    # if not dryrun:
    #     #convert_table = os.path.join()
    #     with open(convert_table, 'a') as fh:
    #         fh.write("%s,%s\n" % (newstim, stim))

    for fileinfo in matchinfo:
        suffix, imgid, ext, filename = fileinfo
        inpath = os.path.join(render_path, filename)
        if suffix == "feedback":
            suffix = "fb"
        newsuffix = (suffix if suffix else "") + (imgid if imgid else "")
        if newsuffix != "":
            newsuffix = "~" + newsuffix

        if ext == ".png":
            outname = newstim + newsuffix + ext
            outpath = os.path.join(image_path, outname)
            if os.path.exists(outpath):
                print "    %s exists" % outname
            else:
                print "    %s --> %s" % (filename, outname)

            if not dryrun:
                shutil.copy(inpath, outpath)

        elif ext == ".avi":
            outnamebase = newstim + newsuffix
            for fmt in formats:
                outname = outnamebase + "." + fmt
                outpath = os.path.join(video_path, outname)
                if os.path.exists(outpath):
                    print "    %s exists" % outname
                else:
                    print "    %s --> %s" % (filename, outname)

                if not dryrun:
                    cmd = cmd_template % (inpath, outpath)
                    subprocess.call(cmd, shell=True)
    # invid = None
    # inimgA = None
    # inimgB = None
    # inimgfloor = None
    # invidfb = None
    # inimgfbA = None
    # inimgfbB = None

    # for f in files:
    #     if f.endswith("-feedback.avi"):
    #         invidfb = f
    #     elif f.endswith("-feedback~A.png"):
    #         inimgfbA = f
    #     elif f.endswith("-feedback~B.png"):
    #         inimgfbB = f
    #     elif f.endswith(".avi"):
    #         invid = f
    #     elif f.endswith("~A.png"):
    #         inimgA = f
    #     elif f.endswith("~B.png"):
    #         inimgB = f
    #     elif f.endswith("-floor.png"):
    #         inimgfloor = f

    # assert invid is not None
    # assert inimgA is not None
    # assert inimgB is not None
    # assert inimgfloor is not None


    # in_path = os.path.join(RENDERPATH, exp_ver)
    # video_path = os.path.join(VIDEOPATH, exp_ver)
    # image_path = os.path.join(IMAGEPATH, exp_ver)

    # invid = os.path.join(in_path, invid)
    # inimgA = os.path.join(in_path, inimgA)
    # inimgB = os.path.join(in_path, inimgB)
    # inimgfloor = os.path.join(in_path, inimgfloor)

    # if invidfb:
    #     invidfb = os.path.join(in_path, invidfb)
    #     inimgfbA = os.path.join(in_path, inimgfbA)
    #     inimgfbB = os.path.join(in_path, inimgfbB)

    # for ext in outext:
    #     outvid = "%s.%s" % (newstim, ext)
    #     outimgA = "%sA.png" % newstim
    #     outimgB = "%sB.png" % newstim
    #     outimgfloor = "%s-floor.png" % newstim

    #     outvid = os.path.join(video_path, outvid)
    #     outimgA = os.path.join(image_path, outimgA)
    #     outimgB = os.path.join(image_path, outimgB)
    #     outimgfloor = os.path.join(image_path, outimgfloor)

    #     if invidfb:
    #         outvidfb = "%s-fb.%s" % (newstim, ext)
    #         outimgfbA = "%s-fbA.png" % newstim
    #         outimgfbB = "%s-fbB.png" % newstim

    #         outvidfb = os.path.join(video_path, outvidfb)
    #         outimgfbA = os.path.join(image_path, outimgfbA)
    #         outimgfbB = os.path.join(image_path, outimgfbB)

    #     if not DRYRUN:
    #         if os.path.exists(outvid):
    #             os.remove(outvid)
    #         if invidfb and os.path.exists(outvidfb):
    #             os.remove(outvidfb)

        # if not DRYRUN:
        #     if not os.path.exists(outvid):
        #         cmd = cmd_template % (invid, outvid)
        #         print "Running '%s'..." % cmd
        #         subprocess.call(cmd, shell=True)
        #     if not os.path.exists(outimgA):
        #         shutil.copy(inimgA, outimgA)
        #     if not os.path.exists(outimgB):
        #         shutil.copy(inimgB, outimgB)
        #     if not os.path.exists(outimgfloor):
        #         shutil.copy(inimgfloor, outimgfloor)

        # if invidfb and not DRYRUN:
        #     if not os.path.exists(outvidfb):
        #         cmd = cmd_template % (invidfb, outvidfb)
        #         print "Running '%s'..." % cmd
        #         subprocess.call(cmd, shell=True)
        #     if not os.path.exists(outimgfbA):
        #         shutil.copy(inimgfbA, outimgfbA)
        #     if not os.path.exists(outimgfbB):
        #         shutil.copy(inimgfbB, outimgfbB)


def loadStimInfo(target):
    stiminfo = {}

    # no previous rendering info exists
    conf_path = os.path.join(CONFPATH, "%s-rendering-info.csv" % target)
    if not os.path.exists(conf_path):
        return stiminfo

    # read csv contents
    with open(conf_path, "r") as fh:
        lines = [x for x in fh.read().split("\n") if x != '']
    # figure out the keys and which one corresponds to 'stimulus'
    keys = lines[0].split(",")
    sidx = keys.index('stimulus')
    del keys[sidx]
    # build a dictionary of stimuli info
    for line in lines[1:]:
        parts = line.split(",")
        stim = parts[sidx]
        del parts[sidx]
        stiminfo[stim] = dict(zip(keys, parts))

    return stiminfo


def getCondition(stim):
    if len(stim.split("~")) == 2:
        condition = []
        cpo, strparams = stim.split("~")
        params = dict([x.split("-", 1) for x in strparams.split("_")])
        if 'kappa' in params:
            num = 10**float(params['kappa'])
            if num == int(num):
                num = int(num)
            condition.append("%s" % num)
        if 'cb' in params:
            condition.append("cb%s" % params['cb'])
        condition = "-".join(condition)
    else:
        condition = None
    return condition


def convert(lists, target, rename, formats, dryrun):

    if dryrun:
        print "DRYRUN -- not actually making changes"

    print "Loading stimulus info..."
    stiminfo = loadStimInfo(target)
    info = {}

    for listname in lists:
        print listname

        # load the stimuli in the list
        listpath = os.path.join(LISTPATH, listname)
        with open(listpath, "r") as fh:
            scenes = [x for x in fh.read().strip().split("\n") if x != ""]

        stims = []
        for scene in scenes:
            if rename is not None:
                stims.append((scene, rename))
            else:
                basename, params = parseStimParams(scene)
                stims.append((scene, basename))

        #         # figure out the new name
        #         # condition = getCondition(oldname)
        #         # suffix = "-%s" % (condition if condition else "")
        #         # newname = basename + suffix
        #         # add the new name to the list of stimuli
        #         stims.append((oldname, listpre))
        #         # # convert it
        #         # print "  %s --> %s" % (oldname, newname)
        #         # convert(oldname, newname)
        #         # # update info dictionary
        #         # for conv in converted:
        #         # info[newname] = stiminfo[oldname].copy()
        #         # info[newname]['example'] = True
        #         # info[newname]['training'] = False

        # else:
        #     for scene in scenes:
        #         # add the scene to the list of stimuli
        #         # # convert it
        #         # print "  %s" % scene
        #         # convert(scene, scene)
        #         # # update info dictionary
        #         # info[scene] = stiminfo[scene].copy()
        #         # info[scene]['example'] = False
        #         # info[scene]['training'] = (listsuff == "training")

        render_path = os.path.join(RENDERPATH, target)
        stims = np.array(stims)
        match = matchStims(stims[:, 0], render_path)

        for oldname, newname in stims:
            matches = match[oldname]
            for m in matches:
                condition = getCondition(m)
                suffix = ("~%s" % condition) if condition else ""
                print "  %s --> %s" % (m, newname + suffix)
                convertStim(m, newname + suffix, matches[m], target,
                            formats, dryrun=dryrun)

                
    # # type conversions -- e.g. make sure integers are ints and not
    # # strings, etc.
    # for i in xrange(len(stims)):
    #     stim = stims[i]
    #     info[stim]['angle'] = int(info[stim]['angle'])
    #     info[stim]['full'] = info[stim]['full'] == 'True'
    #     info[stim]['stable'] = info[stim]['stable'] == 'True'
    #     info[stim]['condition'] = getCondition(stim)
    #     # converted_info[stim] = info[stim]

    # if not dryrun:
    #     # conf_path = os.path.join(CONFPATH, "%s-rendering-info.csv" % target)
    #     filename = "%s-experiment-stimuli.json" % target
    #     conf_path = os.path.join(CONFPATH, filename)
    #     with open(conf_path, "w") as fh:
    #         json.dump(info, fh)

        # filename = "mass-learning-%s-stimuli-converted.json" % exp_ver
        # with open(os.path.join(conf_path, filename), "w") as fh:
        #     json.dump(converted_info, fh)

    # stims = sorted(set([
    #     os.path.splitext(x)[0]
    #     for x in os.listdir(stim_path)
    #     if (x.endswith(".avi") and not
    #         x.endswith("-feedback.avi"))]))


    # # examples
    # for ex, base_label in [
    #         ("stable-example~recording-info.csv", "stable"),
    #         ("unstable-example~recording-info.csv", "unstable"),
    #         ("mass-example~kappa-1.0~recording-info.csv", "mass"),
    #         ("mass-example~kappa--1.0~recording-info.csv", "mass")
    #         ]:
    #     info = parse_info(os.path.join(stim_path, ex))
    #     for key in info.keys():
    #         condition = getCondition(key)
    #         suffix = "-%s" % condition if condition else ""
    #         convert(key, base_label + suffix)
    #         stims.remove(key)

    # # stimuli
    # for i in xrange(len(stims)):
    #     stim = stims[i]
    #     convert(stim, stim, path="stimuli")

    ###############

    # print "Loading info about experiment stimuli..."
    # info_dicts = [
    #     parse_info(os.path.join(
    #         stim_path,
    #         "mass-towers-stability-learning~kappa-1.0~recording-info.csv")),
    #     parse_info(os.path.join(
    #         stim_path,
    #         "mass-towers-stability-learning~kappa--1.0~recording-info.csv")),
    #         ]
    # info = merge_info_dicts(info_dicts)
    # for stim in info:
    #     info[stim]['training'] = False

    # print "Loading info about training stimuli..."
    # info_training_dicts = [
    #     parse_info(os.path.join(
    #         stim_path,
    #         "mass-towers-stability-learning-training~recording-info.csv")),
    #         ]
    # info_training = merge_info_dicts(info_training_dicts)
    # for stim in info_training:
    #     info_training[stim]['training'] = True

    # print "Merging info..."
    # all_info = merge_info_dicts([
    #     parse_info(os.path.join(
    #         conf_path, "mass-learning-%s-stiminfo.csv" % exp_ver)),
    #     info,
    #     info_training])

    # converted_info = {}
    # for i in xrange(len(stims)):
    #     stim = stims[i]
    #     all_info[stim]['angle'] = int(all_info[stim]['angle'])
    #     all_info[stim]['catch'] = all_info[stim]['catch'] == 'True'
    #     all_info[stim]['stable'] = all_info[stim]['stable'] == 'True'
    #     all_info[stim]['condition'] = getCondition(stim)
    #     converted_info[stim] = all_info[stim]

    # if not DRYRUN:
    #     filename = "mass-learning-%s-stimuli.json" % exp_ver
    #     with open(os.path.join(conf_path, filename), "w") as fh:
    #         json.dump(all_info, fh)
    #     filename = "mass-learning-%s-stimuli-converted.json" % exp_ver
    #     with open(os.path.join(conf_path, filename), "w") as fh:
    #         json.dump(converted_info, fh)

###############

if __name__ == "__main__":
    usage = "usage: %prog [options] target list1 [list2 ... listN]"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-r", "--rename", dest="rename", action="store",
        help="rename stimuli to NAME [required]",
        metavar="NAME")
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
    elif len(args) == 1:
        raise ValueError("no stimuli lists passed")
    else:
        target = args[0]
        lists = args[1:]

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
        raise ValueError("no conversion formats specified")

    dryrun = options.dryrun
    rename = options.rename

    convert(lists, target, rename, outext, dryrun)

# outext = ["mp4", "flv", "ogg", "wmv"]

# exp_ver = "E"

# DRYRUN = False
# CONVERT = True

# convert_table = os.path.join(stim_path, "conversion.csv")
# if os.path.exists(convert_table):
#     os.remove(convert_table)
