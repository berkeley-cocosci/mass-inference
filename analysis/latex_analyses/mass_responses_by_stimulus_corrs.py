#!/usr/bin/env python

"""
Produces a LaTeX file with the correlations between model and human mass
responses across stimuli.
"""

__depends__ = ["mass_responses_by_stimulus_corrs.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(
        os.path.join(results_path, "mass_responses_by_stimulus_corrs.csv"))

    results = results\
        .groupby(['version', 'model', 'fitted'])\
        .get_group(('H', 'static', False))\
        .set_index('likelihood')

    latex_pearson = util.load_config()["latex"]["pearson"]

    fh = open(dest, "w")

    for lh, corrs in results.iterrows():
        cmdname = "ExpOneMassRespStimCorr{}".format(
            lh.replace('_', '').capitalize())
        cmd = latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
