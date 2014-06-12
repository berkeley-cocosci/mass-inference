#!/usr/bin/env python

import util
import numpy as np

filename = "num_chance.csv"
texname = "num_chance.tex"

words = [
    'zero', 'one', 'two', 'three', 'four',
    'five', 'six', 'seven', 'eight', 'nine', 'ten'
]


def run(data, results_path, seed):
    np.random.seed(seed)
    results = []

    groups = data['human']['C']\
        .dropna(axis=0, subset=['mass? response'])\
        .groupby('version')\
        .get_group('H')\
        .groupby(['kappa0', 'stimulus'])['mass? correct']
    alpha = 0.05 / len(groups.groups)
    results = groups\
        .apply(util.beta, [alpha])\
        .unstack(-1) <= 0.5

    pth = results_path.joinpath(filename)
    results.to_csv(pth)

    with open(results_path.joinpath(texname), "w") as fh:
        fh.write("%% AUTOMATICALLY GENERATED -- DO NOT EDIT!\n")
        num = results[alpha].sum()
        if num < len(words):
            num = words[num]
        cmd = util.newcommand("NumChance", num)
        fh.write(cmd)

    return pth


if __name__ == "__main__":
    util.run_analysis(run)
