#!/usr/bin/env python

import util

filename = "num_participants.csv"


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
        counts['repeat_worker'])

    pth = results_path.joinpath(filename)
    counts.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
