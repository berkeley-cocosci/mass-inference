#!/usr/bin/env python

import sys
import util


def run(results_path, seed):
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
    util.run_analysis(run, sys.argv[1])
