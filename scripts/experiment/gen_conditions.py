import os
import pickle
import numpy as np

from optparse import OptionParser

CONFDIR = "../../experiment/config"
INFODIR = "../../stimuli/meta"


def loadStiminfo(exp_ver, suffix):
    # load the conversion table
    filename = os.path.join(INFODIR, "%s-conversion.pkl" % exp_ver)
    with open(filename, "r") as fh:
        convtable = pickle.load(fh)
    # load the stimulus info
    filename = os.path.join(INFODIR, "%s-stimulus-info.pkl" % exp_ver)
    with open(filename, "r") as fh:
        stiminfo = pickle.load(fh)

    # convert the names
    convinfo = {}
    for stim in convtable:
        parts = stim.split("~")
        if len(parts) == 1 or parts[1] == suffix:
            convinfo[stim] = stiminfo[convtable[stim]]

    return convinfo


def isTraining(stiminfo):
    val = stiminfo['training'] and not stiminfo['example']
    return val


def isExperiment(stiminfo):
    val = not stiminfo['training'] and not stiminfo['example']
    return val


def createCondition(condition, text_fb, video_fb, qidx, seed,
                    f_training, f_experiment, f_posttest):

    triallist = os.path.join(CONFDIR, "%s_trials.pkl" % condition)
    print "="*70
    print "Creating condition trial list: '%s'" % triallist
    print "-"*70

    if qidx is None:
        qidx = []

    # load stimulus info and find the training and experiment stims
    exp_ver, fbtype, ratio, cb = condition.split("-")
    suffix = "%s-%s" % (ratio, cb)
    stiminfo = loadStiminfo(exp_ver, suffix)
    train0 = sorted([
        stim for stim in stiminfo.keys()
        if isTraining(stiminfo[stim])])
    stims0 = sorted([
        stim for stim in stiminfo.keys()
        if isExperiment(stiminfo[stim])])

    # set up the random number generator
    if seed is not None:
        rso = np.random.RandomState(seed)
    else:
        rso = np.random

    todump = []
    i = 0

    if f_training:
        # random ordering for training
        tidx = np.arange(len(train0))
        rso.shuffle(tidx)
        train = np.array(train0)[tidx]

        # training
        t = 0
        for stim in train:
            si = stiminfo[stim]
            info = {
                'stimulus': stim,
                'trial': t,
                'index': i,
                'ttype': 'training',
                'text_fb': True,
                'video_fb': True,
                'stable': si['stable'],
                'color0': si['color0'],
                'color1': si['color1'],
                'color0_name': si['color0_name'],
                'color1_name': si['color1_name']
                }
            todump.append(info)
            t += 1
            i += 1

    todump.append("finished training")
    i += 1

    if f_experiment:
        # random ordering for experiment
        sidx = np.arange(len(stims0))
        rso.shuffle(sidx)
        stims = np.array(stims0)[sidx]

        # experiment
        t = 0
        for stim in stims:
            si = stiminfo[stim]
            info = {
                'stimulus': stim,
                'trial': t,
                'index': i,
                'ttype': 'experiment',
                'text_fb': text_fb,
                'video_fb': video_fb,
                'stable': si['stable'],
                'color0': si['color0'],
                'color1': si['color1'],
                'color0_name': si['color0_name'],
                'color1_name': si['color1_name']
                }
            todump.append(info)
            t += 1
            i += 1

            if t in qidx and (text_fb or video_fb):
                todump.append("query ratio")
                i += 1

    todump.append("finished experiment")
    i += 1

    if f_posttest:
        # random ordering for posttest
        pidx = np.arange(len(train0))
        rso.shuffle(pidx)
        post = np.array(train0)[pidx]

        # posttest
        t = 0
        for stim in post:
            si = stiminfo[stim]
            info = {
                'stimulus': stim,
                'trial': t,
                'index': i,
                'ttype': 'posttest',
                'text_fb': True,
                'video_fb': True,
                'stable': si['stable'],
                'color0': si['color0'],
                'color1': si['color1'],
                'color0_name': si['color0_name'],
                'color1_name': si['color1_name']
                }
            todump.append(info)
            t += 1
            i += 1

    todump.append("finished posttest")
    i += 1

    for i in todump:
        if isinstance(i, dict):
            print "    " + i['stimulus']
        else:
            print "    " + i

    with open(triallist, "w") as fh:
        pickle.dump(todump, fh)


if __name__ == "__main__":
    usage = "usage: %prog [options] condition"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--text-fb", dest="text_fb", action="store_true",
        default=False, help="show text feedback")
    parser.add_option(
        "--video-fb", dest="video_fb", action="store_true",
        default=False, help="show video feedback")
    parser.add_option(
        "--seed", dest="seed", action="store", type=int,
        default=None, help="pseudorandom number generator seed")
    parser.add_option(
        "--query-trials", dest="qidx", action="store",
        help="CSV list of trials on which to ask about mass ratio")
    parser.add_option(
        "--no-training", dest="f_training", action="store_false",
        default=True, help="don't include training trials")
    parser.add_option(
        "--no-experiment", dest="f_experiment", action="store_false",
        default=True, help="don't include experiment trials")
    parser.add_option(
        "--no-posttest", dest="f_posttest", action="store_false",
        default=True, help="don't include posttest trials")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        raise ValueError("no condition name specified")
    else:
        condition = args[0]

    text_fb = options.text_fb
    video_fb = options.video_fb
    seed = options.seed
    f_training = options.f_training
    f_experiment = options.f_experiment
    f_posttest = options.f_posttest

    if options.qidx is None:
        qidx = np.array([])
    else:
        qidx = np.array([int(x) for x in options.qidx.split(",")])

    createCondition(
        condition,
        text_fb=text_fb,
        video_fb=video_fb,
        seed=seed,
        qidx=qidx,
        f_training=f_training,
        f_experiment=f_experiment,
        f_posttest=f_posttest)


    # create_condition("A-fb", text_fb=True, video_fb=False, seed=0)
    # create_condition("A-nfb", text_fb=False, video_fb=False, seed=0)

    # kwargs = {
    #     'seed': 1,
    #     'qidx': [5, 10, 15, 20]
    #     }
    # create_condition("B-fb-10-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("B-fb-10-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("B-fb-0.1-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("B-fb-0.1-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("B-nfb-10-cb0", text_fb=False, video_fb=False, **kwargs)
    # create_condition("B-nfb-10-cb1", text_fb=False, video_fb=False, **kwargs)

    # kwargs = {
    #     'seed': 2,
    #     'qidx': list(np.round(np.logspace(
    #         np.log10(1), np.log10(40), 8)).astype('i8'))
    #     }
    # create_condition("C-fb-10-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("C-fb-10-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("C-fb-0.1-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("C-fb-0.1-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("C-vfb-10-cb0", text_fb=True, video_fb=True, **kwargs)
    # create_condition("C-vfb-10-cb1", text_fb=True, video_fb=True, **kwargs)
    # create_condition("C-vfb-0.1-cb0", text_fb=True, video_fb=True, **kwargs)
    # create_condition("C-vfb-0.1-cb1", text_fb=True, video_fb=True, **kwargs)
    # create_condition("C-nfb-10-cb0", text_fb=False, video_fb=False, **kwargs)
    # create_condition("C-nfb-10-cb1", text_fb=False, video_fb=False, **kwargs)

    # kwargs = {
    #     'seed': 3,
    #     'qidx': list(np.round(np.logspace(
    #         np.log10(1), np.log10(40), 8)).astype('i8'))
    #     }
    # create_condition("D-fb-10-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("D-fb-10-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("D-fb-0.1-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("D-fb-0.1-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("D-vfb-10-cb0", text_fb=True, video_fb=True, **kwargs)
    # create_condition("D-vfb-10-cb1", text_fb=True, video_fb=True, **kwargs)
    # create_condition("D-vfb-0.1-cb0", text_fb=True, video_fb=True, **kwargs)
    # create_condition("D-vfb-0.1-cb1", text_fb=True, video_fb=True, **kwargs)
    # create_condition("D-nfb-10-cb0", text_fb=False, video_fb=False, **kwargs)
    # create_condition("D-nfb-10-cb1", text_fb=False, video_fb=False, **kwargs)

    # with open("www/config/trial-order-E.txt", "r") as fh:
    #     stims = [x.strip() for x in fh.read().strip().split("\n") if x != ""]
    #     order = np.argsort(np.argsort(stims))

    # kwargs = {
    #     'seed': 4,
    #     'qidx': list(np.round(np.logspace(
    #         np.log10(1), np.log10(40), 8)).astype('i8')),
    #     'order': order
    #     }
    # create_condition("E-fb-10-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("E-fb-10-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("E-fb-0.1-cb0", text_fb=True, video_fb=False, **kwargs)
    # create_condition("E-fb-0.1-cb1", text_fb=True, video_fb=False, **kwargs)
    # create_condition("E-vfb-10-cb0", text_fb=True, video_fb=True, **kwargs)
    # create_condition("E-vfb-10-cb1", text_fb=True, video_fb=True, **kwargs)
    # create_condition("E-vfb-0.1-cb0", text_fb=True, video_fb=True, **kwargs)
    # create_condition("E-vfb-0.1-cb1", text_fb=True, video_fb=True, **kwargs)
    # create_condition("E-nfb-10-cb0", text_fb=False, video_fb=False, **kwargs)
    # create_condition("E-nfb-10-cb1", text_fb=False, video_fb=False, **kwargs)
