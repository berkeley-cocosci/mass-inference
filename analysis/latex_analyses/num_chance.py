#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("num_chance.csv"))

    alpha = results.columns[-1]
    results = results\
        .pivot('stimulus', 'kappa0', alpha)\
        .sum()\
        .sum()

    fh = open(latex_path, "w")
    fh.write(util.newcommand("MassAccNumChance", results))
    fh.write(util.newcommand("MassAccNumChanceCorrection", alpha))
    fh.close()

    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
