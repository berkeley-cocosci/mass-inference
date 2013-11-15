#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from mass import DATA_PATH
from snippets import datapackage as dpkg
import logging
import numpy as np
import pandas as pd
import re
import sys

logger = logging.getLogger('mass.experiment')


def str2bool(x):
    sx = str(x)
    if sx.lower() == 'true':
        return True
    elif sx.lower() == 'false':
        return False
    else:
        return np.nan


def split_uniqueid(df, field):
    workerid, assignmentid = zip(*map(lambda x: x.split(":"), df[field]))
    df['pid'] = workerid
    df['HIT'] = assignmentid
    df = df.drop([field], axis=1)
    return df


def parse_timestamp(df, field):
    timestamp = pd.to_datetime(map(datetime.fromtimestamp, df[field] / 1e3))
    return timestamp


def load_data(data_path, fields):
    data = pd.read_csv(data_path.joinpath(
        "trialdata_all.csv"), header=None)

    data.columns = fields

    data = split_uniqueid(data, 'psiturk_id')

    data['timestamp'] = parse_timestamp(data, 'psiturk_time')
    data['instructions'] = map(str2bool, data['instructions'])
    data['response_time'] = data['response_time'].astype('float') / 1e3
    data['feedback_time'] = data['feedback_time'].astype('float')
    data['presentation_time'] = data['presentation_time'].astype('float')
    data['stable'] = map(str2bool, data['stable'])
    data['nfell'] = data['nfell'].astype('float')
    data['camera_start'] = data['camera_start'].astype('float')
    data['camera_spin'] = data['camera_spin'].astype('float')
    data['response'] = data['response'].astype('float')

    try:
        fall_response = data.groupby('trial_phase').get_group('fall_response')
    except KeyError:
        data['fall? response'] = np.nan
    else:
        data['fall? response'] = fall_response['response']
        data['fall? time'] = fall_response['response_time']

    try:
        mass_response = data.groupby('trial_phase').get_group('mass_response')
    except KeyError:
        data['mass? response'] = np.nan
    else:
        data['mass? response'] = mass_response['response']
        data['mass? time'] = mass_response['response_time']

    data = data\
        .dropna(subset=['response'], how='all')\
        .groupby('instructions')\
        .get_group(False)\
        .rename(columns={
            'index': 'trial',
            'experiment_phase': 'mode'})\
        .drop(['psiturk_time',
               'psiturk_currenttrial',
               'instructions',
               'trial_phase',
               'response',
               'response_time'], axis=1)\
        .set_index(['pid', 'mode', 'trial'])\
        .sortlevel()\
        .reset_index()

    return data


def load_meta(data_path):
    meta = pd.read_csv(data_path.joinpath(
        "questiondata_all.csv"), header=None)
    meta = meta.pivot(index=0, columns=1, values=2)

    conds = split_uniqueid(
        meta[['condition', 'counterbalance']].reset_index(),
        0).set_index('pid')

    meta = meta.drop(['condition', 'counterbalance'], axis=1).drop_duplicates()

    # make sure everyone saw the same questions/possible responses
    assert len(meta) == 1

    exp = re.compile(r" *u{0,1}['\"](.*)['\"] *")
    fields = meta['fields'][0].strip("[]").split(",")
    fields = [exp.search(x).groups() for x in fields]
    for field in fields:
        assert len(field) == 1
    all_fields = ["psiturk_id", "psiturk_currenttrial", "psiturk_time"]
    all_fields.extend([f[0] for f in fields])

    meta = meta.drop(['fields'], axis=1).reset_index(drop=True).T.to_dict()[0]
    meta['participants'] = conds.T.to_dict()

    return meta, all_fields


def load_events(data_path):
    events = pd.read_csv(data_path.joinpath("eventdata_all.csv"))

    events = split_uniqueid(events, 'uniqueid')
    events['timestamp'] = parse_timestamp(events, 'timestamp')
    events = events.set_index(['pid', 'HIT'])\
                   .sortlevel()\
                   .reset_index()

    return events


def save_dpkg(dataset_path, data, meta, events):
    dp = dpkg.DataPackage(name=dataset_path.name, licenses=['odc-by'])
    dp['version'] = '1.0.0'
    dp.add_contributor("Jessica B. Hamrick", "jhamrick@berkeley.edu")
    dp.add_contributor("Thomas L. Griffiths", "tom_griffiths@berkeley.edu")
    dp.add_contributor("Peter W. Battaglia", "pbatt@mit.edu")
    dp.add_contributor("Joshua B. Tenenbaum", "jbt@mit.edu")

    r1 = dpkg.Resource(
        name="experiment.csv", fmt="csv",
        pth="./experiment.csv", data=data)
    r1['mediaformat'] = 'text/csv'
    dp.add_resource(r1)

    r2 = dpkg.Resource(name="experiment_metadata", fmt="json", data=meta)
    r2['mediaformat'] = 'application/json'
    dp.add_resource(r2)

    r3 = dpkg.Resource(
        name="experiment_events.csv", fmt="csv",
        pth="./experiment_events.csv", data=events)
    r3['mediaformat'] = 'text/csv'
    dp.add_resource(r3)

    dp.save(dataset_path.dirname())
    logger.info("Saved to '%s'", dataset_path.relpath())


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force all tasks to be put on the queue.")

    args = parser.parse_args()

    experiment = "mass_inference-G"
    data_path = DATA_PATH.joinpath("human-raw", args.exp)
    dataset_path = DATA_PATH.joinpath("human", "%s.dpkg" % args.exp)

    if dataset_path.exists() and not args.force:
        sys.exit(0)

    if not dataset_path.dirname().exists:
        dataset_path.dirname().makedirs_p()

    meta, fields = load_meta(data_path)
    data = load_data(data_path, fields)
    events = load_events(data_path)
    save_dpkg(dataset_path, data, meta, events)
