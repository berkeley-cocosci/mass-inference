#!/usr/bin/env python

# Builtin
import pickle
import os
import re
from datetime import datetime
# External
import numpy as np
import pandas as pd
import yaml
import json
import dbtools
from path import path
# Cogphysics
import cogphysics
import cogphysics.lib.hashtools as ht
# Local
from snippets import datapackage as dpkg
from mass import DATA_PATH

OLDPATH = DATA_PATH.joinpath("old/old-cogphysics-human-raw-reorganized")
NEWPATH = DATA_PATH.joinpath("human")

convtable_path = os.path.join(
    cogphysics.RESOURCE_PATH,
    "cpobj_conv_stability.pkl")
with open(convtable_path, "r") as fh:
    convtable = pickle.load(fh)
conv_cache = {}


def convert_name(oldname):
    global conv_cache

    if oldname in conv_cache:
        newname = conv_cache[oldname]
    else:
        name1 = ht.forward_find_hashes(oldname)[-1]
        name2 = convtable[name1]

        matches = re.match(r"stability([0-9]+)", name2)
        stimnum = matches.group(1)

        newname = "tower_%s_%s" % (stimnum, "0"*10)
        conv_cache[oldname] = newname
        print "%s --> %s" % (oldname, newname)

    return newname


def process_original(name, data, trial):
    if 'stimulus' in data:
        stim = convert_name(data['stimulus'])
    else:
        stim = convert_name(data['tower_id'])

    if 'start_time' in data:
        time = data['start_time']
    elif 'time' in data:
        time = data['time']
    else:
        time = " ".join(name.split("_")[1:])

    if 'participant_id' in data:
        try:
            pid = "%03d" % data['participant_id']
        except TypeError:
            pid = data['participant_id']
    else:
        pid = name.split("_")[0]

    newdata = {
        'pid': pid,
        'angle': data.get('angle', np.nan),
        'mode': data.get('run_mode', 'experiment'),
        'stimulus': stim,
        'trial': data.get('current_trial', trial),
        'timestamp': time,
    }
    return newdata


def process_stability(name, data, trial):
    newdata = process_original(name, data, trial)

    if 'time_answered' in data:
        if 'raw_answer' in data:
            newdata['fall? response'] = data['raw_answer']
        else:
            newdata['fall? response'] = data['answer']

        newdata['fall? time'] = data['time_answered']
        newdata['camera_spin'] = data.get('camera_spin', True)

        extradata = {
            'question': data['question'][0],
            'possible_responses': data['question'][1]
        }

    else:
        newdata['fall? response'] = data['answer'][0]
        newdata['fall? time'] = data['answer_time']
        newdata['camera_spin'] = data.get('camera_spin', True)

        assert len(data['all_questions']) == 1
        extradata = {
            'question': data['all_questions'][0][1],
            'possible_responses': data['all_questions'][0][2],
        }
    return newdata, extradata


def process_direction(name, data, trial):
    newdata = process_original(name, data, trial)
    newdata['direction? response'] = data['raw_answer']
    newdata['direction? time'] = data['time_answered']

    extradata = {
        'question': data['question'][0],
    }
    return newdata, extradata


def process_direction_discrete(name, data, trial):
    newdata = process_original(name, data, trial)

    if 'raw_answer' not in data:
        newdata['direction? response'] = data['answer']
    else:
        newdata['direction? response'] = data['raw_answer']

    newdata['direction? time'] = data['time_answered']
    newdata['extra_texture_rotation'] = data['extra_texture_rotation']

    extradata = {
        'floor_texture': path(data['extra_texture']).splitpath()[-1],
        'question': data['question'][0],
        'possible_responses': data['question'][1]
    }
    return newdata, extradata


def process_scatter(name, data, trial):
    newdata = process_original(name, data, trial)
    newdata['far? response'] = data['raw_answer']
    newdata['far? time'] = data['time_answered']

    extradata = {
        'question': data['question'][0],
    }
    return newdata, extradata


def process_scatter_discrete(name, data, trial):
    newdata = process_original(name, data, trial)

    if 'raw_answer' not in data:
        newdata['far? response'] = data['answer']
    else:
        newdata['far? response'] = data['raw_answer']

    newdata['far? time'] = data['time_answered']

    extradata = {
        'floor_texture': path(data['extra_texture']).splitpath()[-1],
        'question': data['question'][0],
        'possible_responses': data['question'][1]
    }
    return newdata, extradata


def process_direction_scatter(name, data, trial):
    newdata = process_original(name, data, trial)
    newdata['far? response'] = data['raw_answer'][0]
    newdata['direction? response'] = data['raw_answer'][1]
    newdata['far? time'] = data['time_answered']
    newdata['direction? time'] = data['time_answered']

    extradata = {
        'question': data['question'][0],
    }
    return newdata, extradata


def process_mass(name, data):
    parts = path(name).splitext()[0].split("~")
    pid, expname, mode = parts[0].split("_")
    kappa = float(parts[1].split("-", 1)[1])
    time = " ".join(parts[2].split("_"))

    data['pid'] = pid
    data['mode'] = mode
    data['ratio'] = 10 ** kappa
    data['timestamp'] = time

    for i, stim in enumerate(data.stimulus):
        if stim.startswith("masstower_special"):
            continue
        if stim.startswith("mass-tower_special"):
            continue
        matches = re.search(r"mass-tower_([0-9]+)_([01]+)~.*", stim)
        if matches:
            stimnum = matches.group(1)
            bitstr = matches.group(2)
        else:
            matches = re.search(r"masstower_([0-9]+)_([01]+)", stim)
            if matches:
                stimnum = matches.group(1)
                bitstr = matches.group(2)
            else:
                raise ValueError("can't parse stimulus name")
        newstim = "tower_%s_%s" % (stimnum, bitstr)
        data['stimulus'][i] = newstim

    index = data.set_index(['pid', 'mode', 'trial']).index
    data.index = index
    newdata = data.T.to_dict()
    return newdata


def process_mass_direction(name, data):
    data = data.rename(
        columns={
            'raw_answer': 'direction? response',
            'time_answered': 'direction? time',
        })

    newdata = process_mass(name, data)
    extradata = {
        'question': "In what direction will this tower fall?"
    }

    return newdata, extradata


def process_mass_stability(name, data):
    data = data.rename(
        columns={
            'raw_answer': 'fall? response',
            'time_answered': 'fall? time',
        })

    newdata = process_mass(name, data)
    extradata = {
        'question': "Will this tower fall?",
        'possible_responses': [
            [1, 'definitely unstable'],
            [2, 'probably unstable'],
            [3, 'possible unstable'],
            [4, 'not sure'],
            [5, 'possibly stable'],
            [6, 'probably stable'],
            [7, 'definitely stable']
        ],
    }

    return newdata, extradata


def process_mass_billiards(name, data):
    data = data.rename(
        columns={
            'raw_answer': 'color? response',
            'time_answered': 'color? time',
        })

    parts = path(name).splitext()[0].split("_")
    pid = parts[0]
    mode = parts[2]
    time = " ".join(parts[-2:])

    data['pid'] = pid
    data['mode'] = mode
    data['timestamp'] = time
    data['ratio'] = np.nan

    for i, stim in enumerate(data.stimulus):
        # e.g. pool_triangle_04_1001_d0_8500_d1_1700_v1
        parts = stim.split("_")
        d0 = float(parts[5])
        d1 = float(parts[7])
        ratio = d0 / d1
        data['ratio'][i] = ratio

    index = data.set_index(['pid', 'mode', 'trial']).index
    data.index = index
    newdata = data.T.to_dict()

    extradata = {
        'question': "Which color ball is lighter?",
        'possible_responses': [
            [1, 'definitely red'],
            [2, ''],
            [3, ''],
            [4, "can't tell"],
            [5, ''],
            [6, ''],
            [7, 'definitely yellow']
        ],
    }

    return newdata, extradata


def process_mass_learning(name, data, trials, stims, condition):
    keys = data[0].split(",")

    parts = condition.split("-")
    if len(parts) == 4:
        order, feedback, ratio, counterbalance = parts
        counterbalance = int(counterbalance[2:])
    elif len(parts) == 3:
        order, feedback, ratio = parts
        counterbalance = 0
    elif len(parts) == 2:
        order, feedback = parts
        ratio = 10.0
        counterbalance = 0
    ratio = float(ratio)

    if order == "A":
        timestamp = "2013-01-18"
    elif order == "B":
        timestamp = "2013-01-24"
    elif order == "C":
        timestamp = "2013-01-27"
    elif order == "D":
        timestamp = "2013-01-28"
    elif order == "E":
        timestamp = "2013-01-29"
    elif order == "F":
        timestamp = "2013-04-12"
    else:
        raise ValueError("unhandled order '%s'" % order)

    newdata = {}

    trial = 0
    for lidx, line in enumerate(data[1:]):
        vals = line.split(",")

        if vals[1].startswith("finished"):
            continue

        elif vals[1] == "query ratio":
            trialdata = dict(zip(keys, vals))
            index = int(trialdata['index']) - 1
            newdata[(name, index)]['mass? response'] = trialdata['response']
            newdata[(name, index)]['mass? time'] = trialdata['time']

        else:
            trialdata = dict(zip(keys, vals))
            index = int(trialdata['index'])

            if stims:
                matches = re.match(r"stim_([0-9]+)", trialdata['stimulus'])
                oldstim = "stim_%s" % int(matches.group(1))
                stimulus = stims[oldstim]
            else:
                stimulus = trialdata['stimulus']

            matches = re.match(r"stability([0-9]+)", stimulus)
            if matches:
                num = matches.group(1)
                stimulus = "tower_%s_%s" % (num, "0"*10)
            else:
                matches = re.match(
                    r"mass-tower_([0-9]+)_([01]+).*", stimulus)
                if matches:
                    num = matches.group(1)
                    bitstr = matches.group(2)
                    stimulus = "tower_%s_%s" % (num, bitstr)
                else:
                    raise ValueError("can't parse stimulus name")

            if trialdata['response'] == 'yes':
                response = True
            elif trialdata['response'] == 'no':
                response = False
            else:
                raise ValueError(
                    "unexpected response: %s" % trialdata['response'])

            newdata[(name, index)] = {
                'pid': name,
                'angle': trialdata['angle'],
                'mode': trialdata['ttype'],
                'stimulus': stimulus,
                'trial': trial,
                'fall? response': response,
                'fall? time': trialdata['time'],
                'feedback': feedback,
                'ratio': ratio,
                'counterbalance': counterbalance,
                'order': order,
                'timestamp': timestamp,
            }

            trial += 1

    if newdata == {}:
        newdata[(name, 0)] = {
            'pid': name,
            'trial': 0,
            'ratio': ratio,
            'counterbalance': counterbalance,
            'order': order,
            'timestamp': timestamp,
        }

    extradata = {}
    return newdata, extradata


def merge_ids(exp, turk, metadata):
    CONDITION = ['order', 'feedback', 'ratio', 'counterbalance']

    exp = exp.copy()
    if 'counterbalance' not in exp:
        exp['counterbalance'] = 0
    exp = exp.drop('timestamp', axis=1)

    tt = turk[["WorkerId", "Answer.validation_code", "AcceptTime"]]
    tt.columns = ['worker_id', 'code', 'timestamp']
    tt.timestamp = [datetime
                    .strptime(x, "%a %b %d %H:%M:%S %Z %Y")
                    .isoformat(" ")
                    for x in tt.timestamp]

    hh = pd.DataFrame.from_dict(metadata).reset_index()
    hh.columns = ['pid', 'completion_code', 'validation_code']
    hh = pd.merge(
        hh, exp[['order', 'pid']].drop_duplicates(),
        on='pid', how='outer')

    isA = hh.order == 'A'
    hh['code'] = hh['completion_code']
    hh['code'][isA] = hh['validation_code'][isA]
    hh = hh.drop(['validation_code', 'completion_code'], axis=1)

    ids = pd.merge(
        tt.dropna(subset=['code']), hh,
        on='code',
        how='outer')

    hdata = pd.merge(ids, exp, on=['order', 'pid'], how='outer')

    isnan = np.isnan([1.0 if isinstance(x, str) else x for x in tt.code])
    hdata = hdata.append(tt[isnan]).reset_index(drop=True)

    return hdata

######################################################################

settypes = {
    # stability
    'stability': 'stability',
    'stability_2afc': 'stability',
    'stability_alternating_spin': 'stability',
    'stability_nfb': 'stability',
    'stability_no_spin': 'stability',
    'stability_no_spin_nfb': 'stability',
    'stability_sameheight': 'stability',
    'stability_fmri': 'stability',

    # direction
    'direction_1': 'direction',
    'direction_2': 'direction',
    'direction_discrete_halves': 'direction_discrete',
    'direction_discrete_quadrants': 'direction_discrete',

    # scatter
    'scatter': 'scatter',
    'scatter_sameheight': 'scatter',
    'scatter_discrete': 'scatter_discrete',

    # stability + direction
    'direction_scatter': 'direction_scatter',

    # mass
    'mass_stability_1': 'mass_stability',
    'mass_stability_2': 'mass_stability',
    'mass_direction_1': 'mass_direction',
    'mass_direction_2': 'mass_direction',
    'mass_billiards': 'mass_billiards',

    'mass_learning_A': 'mass_learning',
    'mass_learning_B': 'mass_learning',
    'mass_learning_CD': 'mass_learning',
    'mass_learning_E': 'mass_learning',
    'mass_oneshot_F': 'mass_learning'
}

original_types = [
    'stability',
    'direction',
    'direction_discrete',
    'scatter',
    'scatter_discrete',
    'direction_scatter',
]

mass_prediction_types = [
    'mass_stability',
    'mass_direction',
    'mass_billiards'
]

for dataset in OLDPATH.listdir():
    dataset_name = dataset.splitpath()[-1]
    print dataset_name

    dataset_type = settypes[dataset_name]
    if not dataset_type:
        print "** SKIP"
        continue

    dataset_path = NEWPATH.joinpath(dataset_name + ".dpkg")
    if dataset_path.exists():
        print "** SKIP"
        continue

    if 'nfb' in dataset_name.split("_"):
        feedback = 'nfb'
    else:
        feedback = 'vfb'

    files = dataset.listdir()
    curr_pid = None
    trial = 0

    all_data = {}
    extradata = {}

    for i, filename in enumerate(files):
        name = filename.splitpath()[1]
        print "[%d/%d] %s" % (i, len(files), name)

        if dataset_type in original_types:
            with open(filename, "r") as fh:
                data = yaml.load(fh)

            if 'stimulus' not in data and 'tower_id' not in data:
                continue

            if name.split("_")[0] == curr_pid:
                trial += 1
            else:
                trial = 0
                curr_pid = name.split("_")[0]

            process = locals()['process_%s' % dataset_type]
            newdata, extradata = process(name, data, trial)
            newdata['feedback'] = feedback
            all_data[name] = newdata

        elif dataset_type in mass_prediction_types:
            if not filename.endswith(".csv"):
                continue

            data = pd.DataFrame.from_csv(filename).reset_index()
            data = data.drop('answer', axis=1).rename(
                columns={
                    'current_trial': 'trial',
                    'cam_angle': 'angle'
                })
            data['feedback'] = feedback

            process = locals()['process_%s' % dataset_type]
            newdata, extradata = process(name, data)
            all_data.update(newdata)

        elif dataset_type == 'mass_learning':
            if name == 'stimuli-info.csv':
                continue
            if name.startswith("Batch"):
                continue
            if not filename.endswith(".csv"):
                continue

            with open(filename, "r") as fh:
                data = [x for x in fh.read().split("\n") if x != '']

            trials_path = os.path.join(
                dataset, "%s_trials.json" % name[:-4])
            if trials_path.exists():
                with open(trials_path, "r") as fh:
                    trials = json.load(fh)
            else:
                trials_path = os.path.join(
                    dataset, "%s_trials.pkl" % name[:-4])
                with open(trials_path, "r") as fh:
                    trials = pickle.load(fh)

            stiminfo_path = os.path.join(dataset, "stimuli-info.csv")
            if stiminfo_path.exists():
                with open(stiminfo_path, "r") as fh:
                    stims = [x for x in fh.read().split("\n") if x != '']
                    stims = sorted([x.split(",")[0] for x in stims[1:]])
                    stims = {"stim_%d" % (i+1): x for i, x in enumerate(stims)}
            else:
                stims = None

            # order = dataset.split("_")[-1]
            dbpath = os.path.join(dataset, "data.db")
            tbl = dbtools.Table(dbpath, "Participants")
            condition = tbl[int(name[:-4])].condition.values[0]
            ccode = tbl[int(name[:-4])].completion_code.values[0]
            vcode = tbl[int(name[:-4])].validation_code.values[0]
            if 'completion_codes' not in extradata:
                extradata['completion_codes'] = {}
            extradata['completion_codes'][name[:-4]] = ccode
            if 'validation_codes' not in extradata:
                extradata['validation_codes'] = {}
            extradata['validation_codes'][name[:-4]] = vcode

            process = locals()['process_%s' % dataset_type]
            newdata, extradata_ = process(
                name[:-4], data, trials, stims, condition)
            all_data.update(newdata)

        else:
            raise ValueError("unhandled dataset type: %s" % dataset_type)

    df = pd.DataFrame.from_dict(all_data)
    df = df.stack().unstack(0).set_index(['pid', 'trial', 'stimulus'])
    df = df.reindex_axis(sorted(df.columns), axis=1).reset_index()
    df = df.sort(['pid', 'timestamp', 'trial']).reset_index(drop=True)

    if dataset_type == "mass_learning":
        batches = [pd.DataFrame.from_csv(x) for x in files
                   if x.splitpath()[1].startswith("Batch")]
        batch = pd.concat(batches)
        df = merge_ids(df, batch, extradata)

    dp = dpkg.DataPackage(name=dataset_name + ".dpkg", licenses=['odc-by'])
    dp['version'] = '1.0.0'
    dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
    dp.add_contributor("Peter W. Battaglia", "pbatt@mit.edu")
    dp.add_contributor("Joshua B. Tenenbaum", "jbt@mit.edu")

    r1 = dpkg.Resource(
        name="experiment.csv", fmt="csv",
        pth="./experiment.csv", data=df)
    r1['mediaformat'] = 'text/csv'
    dp.add_resource(r1)

    r2 = dpkg.Resource(name="experiment_metadata", fmt="json", data=extradata)
    r2['mediaformat'] = 'application/json'
    dp.add_resource(r2)

    if dataset_type == "mass_learning":
        dp.add_contributor("Thomas L. Griffiths", "tom_griffiths@berkeley.edu")

        r3 = dpkg.Resource(
            name="turk.csv", fmt="csv",
            pth="./turk.csv", data=batch)
        r3['mediaformat'] = 'text/csv'
        dp.add_resource(r3)

    print "Saving to '%s'..." % dataset_path
    dp.save(NEWPATH)
