#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("fit_mass_responses.csv"))

    results = results.set_index('model')

    fh = open(latex_path, "w")

    for model, corrs in results.iterrows():
        cmdname = "Gamma{}".format(model.capitalize())
        cmd = util.latex_gamma.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
