#!/usr/bin/env python

import sys
import util
import pandas as pd
import numpy as np


def run(latex_path, results_path):
    results = pd.read_csv(
        results_path.joinpath("mass_accuracy_by_participant.csv"))

    results = results.set_index(['version', 'num_mass_trials', 'pid']) 

    replace = {
        'G': 'TwoA',
        'H': 'One',
        'I': 'TwoB',
        8: '',
        20: '',
        1: 'OneTrial',
        2: 'TwoTrials',
        3: 'ThreeTrials',
        4: 'FourTrials',
        5: 'FiveTrials'
    }

    fh = open(latex_path, "w")

    for (version, num_mass_trials), stats in results.groupby(level=['version', 'num_mass_trials']):
        cmdname = "SubjAccExp{}{}Min".format(replace[version], replace[num_mass_trials])
        acc_min = np.round(100 * float(stats.min()), 1)
        if int(acc_min) == acc_min:
            cmd = r"{}\%".format(int(acc_min))
        else:
            cmd = r"{:.1f}\%".format(acc_min)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "SubjAccExp{}{}Max".format(replace[version], replace[num_mass_trials])
        acc_max = np.round(100 * float(stats.max()), 1)
        if int(acc_max) == acc_max:
            cmd = r"{}\%".format(int(acc_max))
        else:
            cmd = r"{:.1f}\%".format(acc_max)
        fh.write(util.newcommand(cmdname, cmd))

        cmdname = "SubjAccExp{}{}Most".format(replace[version], replace[num_mass_trials])
        pct, = np.percentile(stats, [5])
        pct = np.round(100 * float(pct), 1)
        if int(pct) == pct:
            cmd = r"{}\%".format(int(pct))
        else:
            cmd = r"{:.1f}\%".format(pct)
        fh.write(util.newcommand(cmdname, cmd))

    fh.close()
    return latex_path


if __name__ == "__main__":
    util.run_analysis(run, sys.argv[1])
