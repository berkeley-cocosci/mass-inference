#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datetime import datetime
from mass import DATA_PATH
from snippets import datapackage as dpkg
import json
import logging
import numpy as np
import pandas as pd
import sys

logger = logging.getLogger('mass.experiment')


def str2bool(x):
    """Convert a string representation of a boolean (e.g. 'true' or
    'false') to an actual boolean.

    """
    sx = str(x)
    if sx.lower() == 'true':
        return True
    elif sx.lower() == 'false':
        return False
    else:
        return np.nan


def split_uniqueid(df, field):
    """PsiTurk outputs a field which is formatted as
    'workerid:assignmentid'. This function splits the field into two
    separate fields, 'pid' and 'HIT', and drops the old field from the
    dataframe.

    """

    workerid, assignmentid = zip(*map(lambda x: x.split(":"), df[field]))
    df['pid'] = workerid
    df['HIT'] = assignmentid
    df = df.drop([field], axis=1)
    return df


def parse_timestamp(df, field):
    """Parse JavaScript timestamps (which are in millseconds) to pandas
    datetime objects.

    """
    timestamp = pd.to_datetime(map(datetime.fromtimestamp, df[field] / 1e3))
    return timestamp


def load_meta(data_path):
    """Load experiment metadata from the given path. Returns a dictionary
    containing the metadata as well as a list of fields for the trial
    data.

    """
    # load the data and pivot it, so the rows are uniqueid, columns
    # are keys, and values are, well, values
    meta = pd.read_csv(data_path.joinpath(
        "questiondata_all.csv"), header=None)
    meta = meta.pivot(index=0, columns=1, values=2)

    # extract condition information for all participants
    conds = split_uniqueid(
        meta[['condition', 'counterbalance']].reset_index(),
        0).set_index('pid')

    # make sure everyone saw the same questions/possible responses
    meta = meta.drop(['condition', 'counterbalance'], axis=1).drop_duplicates()
    assert len(meta) == 1

    # extract the field names
    fields = ["psiturk_id", "psiturk_currenttrial", "psiturk_time"]
    fields.extend(map(str, json.loads(meta['fields'][0])))

    # convert the remaining metadata to a dictionary and update it
    # with the parsed conditions
    meta = meta.drop(['fields'], axis=1).reset_index(drop=True).T.to_dict()[0]
    meta['participants'] = conds.T.to_dict()

    return meta, fields


def load_data(data_path, fields):
    """Load experiment trial data from the given path. Returns a pandas
    DataFrame.

    """
    # load the data
    data = pd.read_csv(data_path.joinpath(
        "trialdata_all.csv"), header=None)
    # set the column names
    data.columns = fields
    # split apart psiturk_id into pid and HIT
    data = split_uniqueid(data, 'psiturk_id')

    # process other various fields to make sure they're in the right
    # data format
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

    # create separate 'fall? response' / 'fall? time' columns
    try:
        fall_response = data.groupby('trial_phase').get_group('fall_response')
    except KeyError:
        data['fall? response'] = np.nan
    else:
        data['fall? response'] = fall_response['response']
        data['fall? time'] = fall_response['response_time']

    # create separate 'mass? response' / 'mass? time' columns
    try:
        mass_response = data.groupby('trial_phase').get_group('mass_response')
    except KeyError:
        data['mass? response'] = np.nan
    else:
        data['mass? response'] = mass_response['response']
        data['mass? time'] = mass_response['response_time']

    # drop rows that don't have an associated response
    data = data.dropna(subset=['response'], how='all')
    # remove instructions rows
    data = data\
        .groupby('instructions')\
        .get_group(False)

    # rename some columns
    data = data.rename(columns={
        'index': 'trial',
        'experiment_phase': 'mode'})

    # drop columns we don't care about
    data = data.drop([
        'psiturk_time',
        'psiturk_currenttrial',
        'instructions',
        'trial_phase',
        'response',
        'response_time'], axis=1)

    # sort the dataframe by pid/mode/trial
    data = data\
        .set_index(['pid', 'mode', 'trial'])\
        .sortlevel()\
        .reset_index()

    return data


def load_events(data_path):
    """Load experiment event data (e.g. window resizing and the like) from
    the given path. Returns a pandas DataFrame.

    """
    # load the data
    events = pd.read_csv(data_path.joinpath("eventdata_all.csv"))
    # split uniqueid into pid and HIT
    events = split_uniqueid(events, 'uniqueid')
    # parse timestamps
    events['timestamp'] = parse_timestamp(events, 'timestamp')
    # sort by pid/HIT
    events = events\
        .set_index(['HIT', 'pid'])\
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

    # add experiment data, and save it as csv
    r1 = dpkg.Resource(
        name="experiment.csv", fmt="csv",
        pth="./experiment.csv", data=data)
    r1['mediaformat'] = 'text/csv'
    dp.add_resource(r1)

    # add metadata, and save it inline as json
    r2 = dpkg.Resource(name="experiment_metadata", fmt="json", data=meta)
    r2['mediaformat'] = 'application/json'
    dp.add_resource(r2)

    # add event data, and save it as csv
    r3 = dpkg.Resource(
        name="experiment_events.csv", fmt="csv",
        pth="./experiment_events.csv", data=events)
    r3['mediaformat'] = 'text/csv'
    dp.add_resource(r3)

    # save the datapackage
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

    # paths to the data and where we will save it
    data_path = DATA_PATH.joinpath("human-raw", args.exp)
    dest_path = DATA_PATH.joinpath("human", "%s.dpkg" % args.exp)

    # don't do anything if the datapackage already exists
    if dest_path.exists() and not args.force:
        sys.exit(0)

    # create the directory if it doesn't exist
    if not dest_path.dirname().exists:
        dest_path.dirname().makedirs_p()

    # load the data
    meta, fields = load_meta(data_path)
    data = load_data(data_path, fields)
    events = load_events(data_path)

    # save it
    save_dpkg(dest_path, data, meta, events)
