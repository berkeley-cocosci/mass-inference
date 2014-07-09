#!/usr/bin/env python

import util

filename = "condition_counts.csv"


def run(results_path, seed):
    human = util.load_human()

    # compute how many participants we have for each condition
    counts = human['all']\
        .groupby(['version', 'condition', 'counterbalance'])['pid']\
        .apply(lambda x: len(x.unique()))\
        .reset_index()
    counts.columns = [
        'version', 'condition', 'counterbalance', 'num_participants']
    counts = counts.set_index(['version', 'condition', 'counterbalance'])

    pth = results_path.joinpath(filename)
    counts.to_csv(pth)
    return pth


if __name__ == "__main__":
    util.run_analysis(run)
