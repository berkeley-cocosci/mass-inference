#!/usr/bin/env python

import util

filename = "condition_counts.csv"


def run(data, results_path, seed):
    # compute how many participants we have for each condition
    counts = data['exp']\
        .groupby(['condition', 'counterbalance'])['pid']\
        .apply(lambda x: len(x.unique()))\
        .reset_index()
    counts.columns = ['condition', 'counterbalance', 'num_participants']
    counts = counts.set_index(['condition', 'counterbalance'])

    pth = results_path.joinpath(filename)
    counts.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
