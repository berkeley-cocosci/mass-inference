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
        .set_index(['likelihood', 'counterfactual'])

    latex_pearson = util.load_config()["latex"]["pearson"]
    query = util.load_query()

    fh = open(dest, "w")

    for (lh, cf), corrs in results.iterrows():
        if lh == 'ipe_' + query:
            lh = 'Ipe'
        else:
            lh = "".join([x.capitalize() for x in lh.split('_')])

        if cf:
            suffix = "{}CF".format(lh)
        else:
            suffix = "{}NoCF".format(lh)

        cmdname = "ExpOneMassRespStimCorr{}".format(suffix)
        cmd = latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
