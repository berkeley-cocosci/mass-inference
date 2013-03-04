import os, sys, random, subprocess, shutil, yaml, json

cmd_template = "ffmpeg -loglevel error -i %s -r 30 -b 2048k -s 640x480 %s"
outext = ["mp4", "flv", "ogg", "wmv"]

video_path = "www/resources/video"
image_path = "www/resources/images"
stim_path = "stimuli"
conf_path = "www/config"

DRYRUN = False
CONVERT = True

convert_table = os.path.join(stim_path, "conversion.csv")
if os.path.exists(convert_table):
    os.remove(convert_table)
    

def convert(stim, newstim, path="stimuli"):
    files = [x for x in os.listdir(path) if x.startswith(stim)]

    invid = None
    inimgA = None
    inimgB = None
    inimgfloor = None
    invidfb = None
    inimgfbA = None
    inimgfbB = None

    for f in files:
        if f.endswith("-feedback.avi"):
            invidfb = f
        elif f.endswith("-feedback~A.png"):
            inimgfbA = f
        elif f.endswith("-feedback~B.png"):
            inimgfbB = f
        elif f.endswith(".avi"):
            invid = f
        elif f.endswith("~A.png"):
            inimgA = f
        elif f.endswith("~B.png"):
            inimgB = f
        elif f.endswith("-floor.png"):
            inimgfloor = f

    assert invid is not None
    assert inimgA is not None
    assert inimgB is not None
    assert inimgfloor is not None

    if not DRYRUN:
        with open(convert_table, 'a') as fh:
            fh.write("%s,%s\n" % (newstim, os.path.splitext(invid)[0]))

    invid = os.path.join(path, invid)
    inimgA = os.path.join(path, inimgA)
    inimgB = os.path.join(path, inimgB)
    inimgfloor = os.path.join(path, inimgfloor)

    if invidfb: 
        invidfb = os.path.join(path, invidfb)
        inimgfbA = os.path.join(path, inimgfbA)
        inimgfbB = os.path.join(path, inimgfbB)

    for ext in outext:
        outvid = "%s.%s" % (newstim, ext)
        outimgA = "%sA.png" % newstim
        outimgB = "%sB.png" % newstim
        outimgfloor = "%s-floor.png" % newstim

        outvid = os.path.join(video_path, outvid)
        outimgA = os.path.join(image_path, outimgA)
        outimgB = os.path.join(image_path, outimgB)
        outimgfloor = os.path.join(image_path, outimgfloor)

        if invidfb:
            outvidfb = "%s-fb.%s" % (newstim, ext)
            outimgfbA = "%s-fbA.png" % newstim
            outimgfbB = "%s-fbB.png" % newstim

            outvidfb = os.path.join(video_path, outvidfb)
            outimgfbA = os.path.join(image_path, outimgfbA)
            outimgfbB = os.path.join(image_path, outimgfbB)

        # if not DRYRUN:    
        #     if os.path.exists(outvid):
        #         os.remove(outvid)
        #     if invidfb and os.path.exists(outvidfb):
        #         os.remove(outvidfb)
            

        if not DRYRUN:
            if not os.path.exists(outvid):
                cmd = cmd_template % (invid, outvid)
                print "Running '%s'..." % cmd
                subprocess.call(cmd, shell=True)
            if not os.path.exists(outimgA):
                shutil.copy(inimgA, outimgA)
            if not os.path.exists(outimgB):
                shutil.copy(inimgB, outimgB)
            if not os.path.exists(outimgfloor):
                shutil.copy(inimgfloor, outimgfloor)

        if invidfb and not DRYRUN:
            if not os.path.exists(outvidfb):
                cmd = cmd_template % (invidfb, outvidfb)
                print "Running '%s'..." % cmd
                subprocess.call(cmd, shell=True)
            if not os.path.exists(outimgfbA):
                shutil.copy(inimgfbA, outimgfbA)
            if not os.path.exists(outimgfbB):
                shutil.copy(inimgfbB, outimgfbB)

def parse_info(file):
    info = {}
    with open(file, "r") as fh:
        fields = fh.readline().strip().split(",")
        data = [x.split(",") for x in fh.read().strip().split("\n") if x != ""]
        for line in data:
            stim = line[0]
            info[stim] = dict(zip(fields[1:], line[1:]))
    return info

def merge_info_dicts(dicts):
    D = {}
    for d in dicts:
        for key in d:
            if key not in D:
                D[key] = d[key].copy()
            else:
                common = set(D[key].keys()) & set(d[key].keys())
                if not all([D[key][c] == d[key][c] for c in common]):
                    print "key:", key
                    print "common:", common
                    print "old:", D[key]
                    print "new:", d[key]
                    assert False
                D[key].update(d[key])
    return D

def get_condition(stim):
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
    
###############

if DRYRUN:
    print "DRYRUN -- not actually making changes"

stims = sorted(set([
    os.path.splitext(x)[0] 
    for x in os.listdir(stim_path) 
    if (x.endswith(".avi") and not
        x.endswith("-feedback.avi"))]))

# examples
for ex, base_label in [
        ("stable-example~recording-info.csv", "stable"),
        ("unstable-example~recording-info.csv", "unstable"),
        ("mass-example~kappa-1.0~recording-info.csv", "mass"),
        ("mass-example~kappa--1.0~recording-info.csv", "mass")
        ]:
    info = parse_info(os.path.join(stim_path, ex))
    for key in info.keys():
        if CONVERT:
            condition = get_condition(key)
            suffix = "-%s" % condition if condition else ""
            convert(key, base_label + suffix)
        stims.remove(key)
    
# stable_example = parse_info(
#     os.path.join(stim_path, "stable-example~recording-info.csv"))
# unstable_example = parse_info(
#     os.path.join(stim_path, "unstable-example~recording-info.csv"))
# mass_example = parse_info(
#     os.path.join(stim_path, "mass-example~kappa-1.0~recording-info.csv"))

# convert(stable_example.keys()[0], "stable")
# convert(unstable_example.keys()[0], "unstable")
# convert(mass_example.keys()[0], "mass")

###############

#newstims = ["stim_%03d" % (i+1) for i in xrange(len(stims))]
newstims = stims[:]

for i in xrange(len(stims)):
    stim = stims[i]
    newstim = newstims[i]
    if CONVERT:
        convert(stim, newstim, path="stimuli")

###############

print "Loading info about experiment stimuli..."
info_dicts = [
    parse_info(os.path.join(
        stim_path, 
        "mass-towers-stability-learning~kappa-1.0~recording-info.csv")),
    parse_info(os.path.join(
        stim_path, 
        "mass-towers-stability-learning~kappa--1.0~recording-info.csv")),
    ]
info = merge_info_dicts(info_dicts)
for stim in info:
    info[stim]['training'] = False

print "Loading info about training stimuli..."
info_training_dicts = [
    parse_info(os.path.join(
        stim_path, "mass-towers-stability-learning-training~recording-info.csv")),
]
info_training = merge_info_dicts(info_training_dicts)
for stim in info_training:
    info_training[stim]['training'] = True

print "Merging info..."
all_info = merge_info_dicts([
    parse_info(os.path.join(conf_path, "stimuli-info.csv")),
    info, 
    info_training])

converted_info = {}
for i in xrange(len(stims)):
    stim = stims[i]
    all_info[stim]['angle'] = int(all_info[stim]['angle'])
    all_info[stim]['catch'] = all_info[stim]['catch'] == 'True'
    all_info[stim]['stable'] = all_info[stim]['stable'] == 'True'
    all_info[stim]['condition'] = get_condition(stim)
    converted_info[newstims[i]] = all_info[stim]

if not DRYRUN:
    with open(os.path.join(conf_path, "stimuli.json"), "w") as fh:
        json.dump(all_info, fh)
    with open(os.path.join(conf_path, "stimuli-converted.json"), "w") as fh:
        json.dump(converted_info, fh)



