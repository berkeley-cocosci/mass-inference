#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("mass_accuracy_by_stimulus_corrs.csv"))

    results = results.set_index(['version', 'X', 'Y'])

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    fh = open(latex_path, "w")

    for (version, x, y), corrs in results.iterrows():
        cmdname = "Exp{}MassAccStimCorr{}".format(
            replace[version], x)
        cmd = util.latex_spearman.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
