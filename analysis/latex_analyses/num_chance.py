#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("num_chance.csv"))

    alpha = results.columns[-1]
    results = results\
        .groupby('version')\
        .sum()[alpha]

    replace = {
        'H': 'One',
        'G': 'TwoA',
        'I': 'TwoB'
    }

    fh = open(latex_path, "w")

    for version, num in results.iteritems():
        name = "MassAccNumChanceExp{}".format(replace[version])
        fh.write(util.newcommand(name, int(num)))
        fh.write(util.newcommand("{}Correction".format(name), alpha))

    fh.close()

    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
