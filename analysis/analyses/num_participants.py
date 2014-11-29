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

import util


def run(results_path):
    participants = util.load_participants()

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

    counts.to_csv(results_path)
    return results_path


if __name__ == "__main__":
    parser = util.default_argparser(__doc__, add_seed=False)
    args = parser.parse_args()
    run(args.results_path)
