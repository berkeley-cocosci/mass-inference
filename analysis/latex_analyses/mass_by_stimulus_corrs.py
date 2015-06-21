#!/usr/bin/env python

"""
Produces a LaTeX file with the correlations between model and human mass
accuracy across stimuli.
"""

__depends__ = [
    "mass_accuracy_by_stimulus_corrs.csv",
    "mass_responses_by_stimulus_corrs.csv",
    "mass_by_stimulus_human_corrs.csv"
]

import os
import util
import pandas as pd


def run(dest, results_path):
    accuracy = pd\
        .read_csv(os.path.join(results_path, "mass_accuracy_by_stimulus_corrs.csv"))\
        .groupby('version')\
        .get_group('H')\
        .set_index(['likelihood', 'counterfactual'])

    responses = pd\
        .read_csv(os.path.join(results_path, "mass_responses_by_stimulus_corrs.csv"))\
        .groupby('version')\
        .get_group('H')\
        .set_index(['likelihood', 'counterfactual'])

    human = pd\
        .read_csv(os.path.join(results_path, "mass_by_stimulus_human_corrs.csv"))\
        .set_index('judgment')\
        .T

    latex_pearson = util.load_config()["latex"]["pearson"]
    query = util.load_query()

    fh = open(dest, "w")

    for (lh, cf), corrs in accuracy.iterrows():
        if lh == 'ipe_' + query:
            lh = 'Ipe'
        else:
            lh = "".join([x.capitalize() for x in lh.split('_')])

        if cf:
            suffix = "{}CF".format(lh)
        else:
            suffix = "{}NoCF".format(lh)

        cmdname = "ExpOneMassAccStimCorr{}".format(suffix)
        cmd = latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    for (lh, cf), corrs in responses.iterrows():
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

    fh.write(util.newcommand("MassRespHumanCorr", latex_pearson.format(**human['mass? response'])))
    fh.write(util.newcommand("MassAccHumanCorr", latex_pearson.format(**human['mass? correct'])))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
