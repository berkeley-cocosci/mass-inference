#!/usr/bin/env python

"""
Compute the number of participants in each condition, and outputs a csv with
the following columns:

    version (string)
        the experiment version
    duplicate_trials (int)
        how many participants had duplicate trials
    failed_posttest (int)
        how many participants failed the posttest
    incomplete (int)
        how many participants failed to finish
    ok (int)
        how many participants are ok
    repeat_worker (int)
        how many participants did a previous version of the experiment
    complete (int)
        how many participants completed the experiment
    excluded (int)
        how many participants were excluded from analysis

"""

__depends__ = ["human"]

import util


def run(dest, data_path):
    participants = util.load_participants(data_path)

    counts = participants\
        .fillna('ok')\
        .groupby(['version', 'note'])\
        .apply(lambda x: len(x))\
        .unstack('note')\
        .fillna(0)

    counts['complete'] = (
        counts['failed_posttest'] +
        counts['ok'] +
        counts['repeat_worker'] +
        counts['duplicate_trials'])

    counts['excluded'] = (
        counts['failed_posttest'] +
        counts['repeat_worker'] +
        counts['duplicate_trials'])

    counts.to_csv(dest)
    return dest


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.data_path)
