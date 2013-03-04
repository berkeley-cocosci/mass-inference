import os
import json
import numpy as np

F_TRAINING = True
F_EXPERIMENT = True
F_POSTTEST = True

CONF_DIR = "www/config"
INFO_DIR = "../stimuli"


def get_all_stiminfo(idx):
    filename = os.path.join(INFO_DIR, "stimuli-converted.json")
    with open(filename, "r") as fh:
        all_stiminfo = json.load(fh)
    stiminfo = {}
    suffix = "-".join(idx.split("-")[-2:])
    for stim in all_stiminfo:
        if (all_stiminfo[stim]['training'] or
            all_stiminfo[stim]['condition'] == suffix):
            stiminfo[stim] = all_stiminfo[stim]
    return stiminfo


def create_condition(idx, text_fb, visual_fb, qidx=None, seed=None, order=None):
    triallist = os.path.join(CONF_DIR, "%s_trials.json" % idx)
    print "Creating condition trial list: '%s'" % triallist

    if qidx is None:
        qidx = []

    stiminfo = get_all_stiminfo(idx)
    train0 = sorted([stim for stim in stiminfo.keys()
                     if stiminfo[stim]['training']])
    stims0 = sorted([stim for stim in stiminfo.keys()
                     if not stiminfo[stim]['training']])

    if seed is not None:
        rso = np.random.RandomState(seed)
    else:
        rso = np.random

    todump = []
    i = 0

    if F_TRAINING:
        # random ordering for training
        tidx = np.arange(len(train0))
        rso.shuffle(tidx)
        train = np.array(train0)[tidx]

        # training
        t = 0
        for stim in train:
            info = stiminfo[stim].copy()
            info.update(stimulus=stim, trial=t, index=i, ttype='training',
                        text_fb=True, visual_fb=True)
            del info['training']
            del info['catch']
            todump.append(info)
            t += 1
            i += 1

    todump.append("finished training")
    i += 1

    if F_EXPERIMENT:
        # random ordering for experiment
        sidx = np.arange(len(stims0))
        rso.shuffle(sidx)
        if order is not None:
            sidx = order
        stims = np.array(stims0)[sidx]

        # experiment
        t = 0
        for stim in stims:
            info = stiminfo[stim].copy()
            info.update(stimulus=stim, trial=t, index=i, ttype='experiment',
                        text_fb=text_fb, visual_fb=visual_fb)
            del info['training']
            del info['catch']
            todump.append(info)

            t += 1
            i += 1

            if t in qidx and (text_fb or visual_fb):
                todump.append("query ratio")
                i += 1

    todump.append("finished experiment")
    i += 1

    if F_POSTTEST:
        # random ordering for posttest
        pidx = np.arange(len(train0))
        rso.shuffle(pidx)
        post = np.array(train0)[pidx]

        # posttest
        t = 0
        for stim in post:
            info = stiminfo[stim].copy()
            info.update(stimulus=stim, trial=t, index=i, ttype='posttest',
                        text_fb=True, visual_fb=True)
            del info['training']
            del info['catch']
            todump.append(info)
            t += 1
            i += 1

    todump.append("finished posttest")
    i += 1

    for i in todump:
        if isinstance(i, dict):
            print i['stimulus']
        else:
            print i

    with open(triallist, "w") as fh:
        json.dump(todump, fh)

if __name__ == "__main__":
    # create_condition("A-fb", text_fb=True, visual_fb=False, seed=0)
    # create_condition("A-nfb", text_fb=False, visual_fb=False, seed=0)

    # kwargs = {
    #     'seed': 1,
    #     'qidx': [5, 10, 15, 20]
    #     }
    # create_condition("B-fb-10-cb0", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("B-fb-10-cb1", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("B-fb-0.1-cb0", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("B-fb-0.1-cb1", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("B-nfb-10-cb0", text_fb=False, visual_fb=False, **kwargs)
    # create_condition("B-nfb-10-cb1", text_fb=False, visual_fb=False, **kwargs)

    # kwargs = {
    #     'seed': 2,
    #     'qidx': list(np.round(np.logspace(
    #         np.log10(1), np.log10(40), 8)).astype('i8'))
    #     }
    # create_condition("C-fb-10-cb0", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("C-fb-10-cb1", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("C-fb-0.1-cb0", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("C-fb-0.1-cb1", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("C-vfb-10-cb0", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("C-vfb-10-cb1", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("C-vfb-0.1-cb0", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("C-vfb-0.1-cb1", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("C-nfb-10-cb0", text_fb=False, visual_fb=False, **kwargs)
    # create_condition("C-nfb-10-cb1", text_fb=False, visual_fb=False, **kwargs)

    # kwargs = {
    #     'seed': 3,
    #     'qidx': list(np.round(np.logspace(
    #         np.log10(1), np.log10(40), 8)).astype('i8'))
    #     }
    # create_condition("D-fb-10-cb0", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("D-fb-10-cb1", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("D-fb-0.1-cb0", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("D-fb-0.1-cb1", text_fb=True, visual_fb=False, **kwargs)
    # create_condition("D-vfb-10-cb0", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("D-vfb-10-cb1", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("D-vfb-0.1-cb0", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("D-vfb-0.1-cb1", text_fb=True, visual_fb=True, **kwargs)
    # create_condition("D-nfb-10-cb0", text_fb=False, visual_fb=False, **kwargs)
    # create_condition("D-nfb-10-cb1", text_fb=False, visual_fb=False, **kwargs)

    with open("www/config/trial-order-E.txt", "r") as fh:
        stims = [x.strip() for x in fh.read().strip().split("\n") if x != ""]
        order = np.argsort(np.argsort(stims))

    kwargs = {
        'seed': 4,
        'qidx': list(np.round(np.logspace(
            np.log10(1), np.log10(40), 8)).astype('i8')),
        'order': order
        }
    create_condition("E-fb-10-cb0", text_fb=True, visual_fb=False, **kwargs)
    create_condition("E-fb-10-cb1", text_fb=True, visual_fb=False, **kwargs)
    create_condition("E-fb-0.1-cb0", text_fb=True, visual_fb=False, **kwargs)
    create_condition("E-fb-0.1-cb1", text_fb=True, visual_fb=False, **kwargs)
    create_condition("E-vfb-10-cb0", text_fb=True, visual_fb=True, **kwargs)
    create_condition("E-vfb-10-cb1", text_fb=True, visual_fb=True, **kwargs)
    create_condition("E-vfb-0.1-cb0", text_fb=True, visual_fb=True, **kwargs)
    create_condition("E-vfb-0.1-cb1", text_fb=True, visual_fb=True, **kwargs)
    create_condition("E-nfb-10-cb0", text_fb=False, visual_fb=False, **kwargs)
    create_condition("E-nfb-10-cb1", text_fb=False, visual_fb=False, **kwargs)
