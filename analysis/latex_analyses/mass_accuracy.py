#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("mass_accuracy.csv"))

    results = results\
        .set_index(['species', 'class', 'version', 'kappa0'])\
        .ix[('human', 'static')]

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three',
        '-1.0': 'KappaLow',
        '1.0': 'KappaHigh',
        'all': ''
    }

    fh = open(latex_path, "w")

    for (version, kappa0), accuracy in results.iterrows():
        cmdname = "MassAccExp{}{}".format(
            replace[version], replace[kappa0])
        cmd = util.latex_percent.format(**accuracy)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
