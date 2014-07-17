#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("fall_response_corrs.csv"))

    results = results.set_index(['block', 'X', 'Y'])

    fh = open(latex_path, "w")

    for (block, x, y), corrs in results.iterrows():
        cmdname = "FallCorr{}{}{}".format(x, y, block)
        cmd = util.latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
