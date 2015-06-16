#!/usr/bin/env python

"""
Produces a LaTeX file with the correlations between model and human mass
responses across stimuli.
"""

__depends__ = ["precision_recall.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(
        os.path.join(results_path, "precision_recall.csv"))

    results = results\
        .groupby(['version', 'model', 'fitted'])\
        .get_group(('H', 'static', False))\
        .set_index(['likelihood', 'counterfactual'])

    fh = open(dest, "w")

    for (lh, cf), res in results.iterrows():
        if cf:
            suffix = "{}CF".format(lh.replace('_', '').capitalize())
        else:
            suffix = "{}NoCF".format(lh.replace('_', '').capitalize())

        cmdname = "ExpOneFScore{}".format(suffix)
        cmd = "F_1={F1:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "ExpOnePrecision{}".format(suffix)
        cmd = "{precision:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "ExpOneRecall{}".format(suffix)
        cmd = "{recall:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "ExpOneAccuracy{}".format(suffix)
        cmd = r"{:.1f}\%".format(res['accuracy'] * 100)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.to, args.results_path)
