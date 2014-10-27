#!/usr/bin/env python

import sys
import util
import pandas as pd


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("mass_accuracy_by_trial_corrs.csv"))

    results = results\
        .set_index(['kappa0', 'version', 'num_mass_trials'])\
        .ix['all']\
        .ix[[('G', 8), ('H', 20), ('I', -1), ('I', 5)]]

    replace = {
        'H': 'One',
        'G': 'Two',
        'I': 'Three'
    }

    fh = open(latex_path, "w")

    for (version, num), corrs in results.iterrows():
        if version == 'I' and num == 5:
            cmdname = "Exp{}MassTrialCorrWithinSubjs".format(
                replace[version])
        else:
            cmdname = "Exp{}MassTrialCorr".format(
                replace[version])
        cmd = util.latex_pearson.format(**corrs)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
