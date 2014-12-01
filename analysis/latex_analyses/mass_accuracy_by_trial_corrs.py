#!/usr/bin/env python

"""
Produces a LaTeX file with the correlations between trial number and human mass
accuracy.
"""

__depends__ = ["mass_accuracy_by_trial_corrs.csv"]

import os
import util
import pandas as pd


def run(dest, results_path):
    results = pd.read_csv(
        os.path.join(results_path, "mass_accuracy_by_trial_corrs.csv"))

    results = results\
        .set_index(['version', 'num_mass_trials'])\
        .ix[[('G', 8), ('H', 20), ('I', -1), ('I', 5)]]

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    latex_pearson = util.load_config()["latex"]["pearson"]

    fh = open(dest, "w")

    for (version, num), corrs in results.iterrows():
        if version == 'I' and num == 5:
            cmdname = "Exp{}MassTrialCorrWithinSubjs".format(
                replace[version])
        else:
            cmdname = "Exp{}MassTrialCorr".format(
                replace[version])
        cmd = latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return dest


if __name__ == "__main__":
    parser = util.default_argparser(locals())
    args = parser.parse_args()
    run(args.dest, args.results_path)
