#!/usr/bin/env python

"""
Produces a LaTeX file with the correlations between model and human mass
responses across stimuli.
"""

__depends__ = ["precision_recall.csv"]

import os
import util
import pandas as pd


def run(dest, results_path, counterfactual):
    results = pd.read_csv(
        os.path.join(results_path, "precision_recall.csv"))

    results = results\
        .groupby(['version', 'model', 'counterfactual', 'fitted'])\
        .get_group(('H', 'static', counterfactual, False))\
        .set_index('likelihood')

    fh = open(dest, "w")

    for lh, res in results.iterrows():
        cmdname = "ExpOneFScore{}".format(lh.replace('_', '').capitalize())
        cmd = "F_1={F1:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "ExpOnePrecision{}".format(lh.replace('_', '').capitalize())
        cmd = "{precision:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "ExpOneRecall{}".format(lh.replace('_', '').capitalize())
        cmd = "{recall:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "ExpOneAccuracy{}".format(lh.replace('_', '').capitalize())
        cmd = "{accuracy:.2f}".format(**res)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()


if __name__ == "__main__":
    config = util.load_config()
    parser = util.default_argparser(locals())
    if config['analysis']['counterfactual']:
        parser.add_argument(
            '--no-counterfactual',
            action='store_false',
            dest='counterfactual',
            default=True,
            help="don't plot the counterfactual likelihoods")
    else:
        parser.add_argument(
            '--counterfactual',
            action='store_true',
            dest='counterfactual',
            default=False,
            help='plot the counterfactual likelihoods')
    args = parser.parse_args()
    run(args.to, args.results_path, args.counterfactual)
